#!/usr/bin/env python3
"""
Provides the code for testing the main capabilities of the Blade damage project

Code to compute the generated moments and forces due to propeller damage
Assumptions:
- Homogeneous mass along the blade: the centroid equals the location of the cg
- The blade is cut parallel to the edge of the propeller such that the remaining polygon is still a trapezoid
- The Bebop 2 blades are simplified as two trapezoids with parallel sides connected by the long parallel side
- The induced velocity is computed with the simplified linear induced inflow
- The nonlinear effects between (damaged) blades are not considered
- The data used for the cl cd identification is obtained from the Matlab model that provides the propeller thrust
- The airfoil is constant throughout the blade
- The linear induced inflow is a good approximation of the real induced velocity
- The cross flow is ignored, along the span of the blade
# TODO: try different solver
# TODO: consider removing added code in the input generator
# TODO: repair angle printing function

Recommendations:
- Creation of a black-box model that provides the highly-nonlinear lift and drag contributions of each blade section
that are not encapsulated in the BEM model. For instance, the interaction between propellers and the interaction
between the propeller and the body. Work similar to efforts in CFD to accelerate the run of CFD with a simple easy to
compute model and a nonlinear black-box model on top. It is true that there is already a black-box model for the whole
propeller, but this would be interesting for the failure scenarios in which blade section information is required.
In this black-box model for nonlinearities, it would be interesting to also train with data from broken propellers. Then
an input to the black-box model would be the percentage of broken blade in the propeller.
"""

# TODO: create tests for classes and methods in helper_func_unittest.py

# Modules to import
from helper_func import *
from user_input import *
from Propeller import Propeller
from helper_func import multi_figure_storage, plot_coeffs_map
import matplotlib.pyplot as plt

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"

# Create the propeller and the blades
propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, state_blades, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

# ----------------------------------------------------------------------------------------------------------------------
# Compute the location of the center of gravity of the propeller and the BladeSection chords
cg_location = propeller.compute_cg_location()
average_chords, segment_chords = compute_average_chords(chord_lengths_rt_lst, length_trapezoids_rt_lst, n_blade_segment_lst[0])
print(average_chords)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the forces and moments generated by the shift in the center of gravity
# Local input
omega_local = 500
time = 1
dt = 0.001

# Computations
n_points = int(time/dt + 1)
F_lst = np.zeros((3, n_points))
M_lst = np.zeros((3, n_points))
rotation_angle_lst = np.zeros(n_points)
rotation_angle = 0
for i in range(n_points):
    F, M = propeller.compute_cg_forces_moments(omega_local, attitude)
    F_lst[:, i] = F.flatten()
    M_lst[:, i] = M.flatten()
    rotation_angle_lst[i] = rotation_angle
    rotation_angle = propeller.update_rotation_angle(omega_local, dt)
# plot_FM(np.arange(0, time+dt, dt), rotation_angle_lst, F_lst, M_lst)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the lift/thrust generated by the propeller
T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega)
print(T)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the cl-alpha curve polynomial coefficients
if coefficients_identification:
    coeffs_grid = np.zeros((len(n_blade_segment_lst), len(number_samples_lst), degree_cla+degree_cda+2))
    for i, n_blade_segment in enumerate(n_blade_segment_lst):
        for j, number_samples in enumerate(number_samples_lst):
            coeffs, A, b = propeller.compute_cla_coeffs(number_samples, n_blade_segment, degree_cla, degree_cda,
                                                        min_w=min_w, max_w=max_w, va=va, rho=1.225,
                                                        activate_plotting=True,
                                                        activate_params_blade_contribution_plotting=
                                                        activate_params_blade_contribution_plotting,
                                                        LS_method=LS_method, start_plot=start_cla_plot,
                                                        finish_plot=finish_cla_plot,
                                                        switch_avg_rot=switch_avg_rot, n_rot_steps=n_rot_steps,
                                                        optimization_method=optimization_method,
                                                        min_method=min_method, switch_constrains=switch_constrains)
            coeffs_grid[i, j, :] = np.reshape(coeffs, [-1, ])
            filename = f'Saved_figures/{number_samples}_dp_{n_blade_segment}_bs_{va}_va_{min_method}.pdf'
            multi_figure_storage(filename, figs=None, dpi=200)
            plt.close('all')
            cla_coeffs = coeffs[:degree_cla + 1, 0]
            cda_coeffs = coeffs[degree_cla + 1:, 0]
            print(cla_coeffs)
            print(cda_coeffs)
plot_coeffs_map(coeffs_grid, degree_cla, degree_cda)
filename = f'Saved_figures/{n_blade_segment_lst[0]}_{n_blade_segment_lst[-1]}_dp_{n_blade_segment_lst[0]}_{n_blade_segment_lst[-1]}_bs_{va}_va_{min_method}.pdf'
multi_figure_storage(filename, figs=None, dpi=200)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the thrust, thrust moment, x-y in plane force and torque
# Local input
cla_coeffs = np.array([-4.65298061,   187.59293314, -1413.02494156,  3892.87107255, -4410.25625439,  1744.50538986])
cda_coeffs = np.array([-1.66217226e+00, -6.65648057e+01,  7.26049638e+02, -1.84544413e+03, 1.06106349e+03,  4.11807861e+02])
omega = 1256          # [rad/s]
propeller_speed = np.array([[np.sqrt(24)], [0], [-1]])

# Computations
T_remaining, T_damaged, M_remaining, M_damaged = propeller.compute_thrust_moment(n_blade_segment, omega,
                                                                                 propeller_speed, cla_coeffs,
                                                                                 cda_coeffs)
Q_remaining, Q_damaged, F_remaining, F_damaged = propeller.compute_torque_force(n_blade_segment, omega, propeller_speed,
                                                                                cla_coeffs, cda_coeffs)
print(T_remaining, T_damaged, M_remaining, M_damaged)
print(Q_remaining, Q_damaged, F_remaining, F_damaged)

# ----------------------------------------------------------------------------------------------------------------------
# Put all the forces and moments together
# Local input
propeller.set_rotation_angle(0)
cla_coeffs = np.array([-5.89,   251.29, -1919.48,  5298.43, -5990.51,  2364.58])
cda_coeffs = np.array([-1.39, -119.59,  1348.14, -4142.33, 4331.41,  -1157.51])
omega_local = 500       # [rad/s]
time = 1
dt = 0.001

# Computations
n_points = int(time/dt + 1)
F_lst = np.zeros((3, n_points))
M_lst = np.zeros((3, n_points))
rotation_angle_lst = np.zeros(n_points)
rotation_angle = 0
for i in range(n_points):
    if not i % 10:
        print(f'Iteration {i} out of {n_points-1}')
    F = np.zeros((3, 1))
    M = np.zeros((3, 1))

    # Computation of forces and moments that derive from the change in mass
    F_cg, M_cg = propeller.compute_cg_forces_moments(omega_local, attitude)
    F += F_cg
    M += M_cg

    # Computation of moments and forces derived from the loss in an aerodynamic surface
    T_remaining, T_damaged, M_remaining, M_damaged = propeller.compute_thrust_moment(n_blade_segment, omega_local,
                                                                                     propeller_speed, cla_coeffs,
                                                                                     cda_coeffs)
    Q_remaining, Q_damaged, F_remaining, F_damaged = propeller.compute_torque_force(n_blade_segment, omega_local,
                                                                                    propeller_speed, cla_coeffs,
                                                                                    cda_coeffs)
    F[2, 0] -= T_damaged
    M -= M_damaged.T
    M[2, 0] -= Q_damaged
    F -= F_damaged.T

    F_lst[:, i] = F.flatten()
    M_lst[:, i] = M.flatten()
    rotation_angle_lst[i] = rotation_angle
    rotation_angle = propeller.update_rotation_angle(omega_local, dt)

# Plot the forces and moments
plot_FM(np.arange(0, time+dt, dt), rotation_angle_lst, F_lst, M_lst)

