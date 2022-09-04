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
- The twist decreases linearly from the root to the tip
# TODO: try different solver

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
from helper_func import multi_figure_storage, plot_coeffs_map, compute_coeffs_grid_row, store_Abu_data
import matplotlib.pyplot as plt
import pickle
import time

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"

# Create the propeller and the blades
propeller = Propeller(propeller_number, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=switch_chords_twist_plotting)
propeller.create_blades()

# ----------------------------------------------------------------------------------------------------------------------
# Compute the location of the center of gravity of the propeller and the BladeSection chords
cg_location = propeller.compute_cg_location()
average_chords, segment_chords = compute_average_chords(chord_lengths_rt_lst, length_trapezoids_rt_lst, n_blade_segment_lst[0])
print(average_chords)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the forces and moments generated by the shift in the center of gravity
# Computations
if switch_plot_mass:
    n_points = int(total_time/dt + 1)
    F_lst_mass = np.zeros((3, n_points))
    M_lst_mass = np.zeros((3, n_points))
    rotation_angle_lst = np.zeros(n_points)
    rotation_angle = 0
    propeller.set_rotation_angle(rotation_angle)
    for i in range(n_points):
        F, M = propeller.compute_cg_forces_moments(omega, attitude)
        F_lst_mass[:, i] = F.flatten()
        M_lst_mass[:, i] = M.flatten()
        rotation_angle_lst[i] = rotation_angle
        rotation_angle = propeller.update_rotation_angle(omega, dt)
    plot_FM(np.arange(0, total_time+dt, dt), rotation_angle_lst, F_lst_mass, M_lst_mass, mass_aero="m")

# ----------------------------------------------------------------------------------------------------------------------
# Compute the lift/thrust generated by the propeller
T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega)
# print(T)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the cl-alpha curve polynomial coefficients
if coefficients_identification:
    if switch_coeffs_grid_plot:
        coeffs_grid = np.zeros((len(n_blade_segment_lst), len(number_samples_lst), degree_cla+degree_cda+2))
        activate_plotting = False
        warm_starts = np.array([ 2.91490303e-01,  4.49088438e+00, -1.16254588e+01,  9.45445197e-03,
                                 -8.12536151e-01,  1.55181623e+01])
    else:
        activate_plotting = True
    for i, n_blade_segment in enumerate(n_blade_segment_lst):
        number_samples = number_samples_lst[-1]

        # Check the number of samples already available
        if switch_recycle_samples:
            available_samples, samples_to_compute = \
                check_Abu_data(n_blade_segment, number_samples, va, min_method, file_name_suffix)
        else:
            samples_to_compute = number_samples

        if samples_to_compute != 0:
            now = time.time()
            coeffs, A, b, u = propeller.compute_cla_coeffs(samples_to_compute, n_blade_segment, degree_cla, degree_cda,
                                                           min_w=min_w, max_w=max_w, va=va, rho=1.225,
                                                           activate_plotting=activate_plotting,
                                                           activate_params_blade_contribution_plotting=
                                                           activate_params_blade_contribution_plotting,
                                                           LS_method=LS_method, start_plot=start_cla_plot,
                                                           finish_plot=finish_cla_plot,
                                                           switch_avg_rot=switch_avg_rot, n_rot_steps=n_rot_steps,
                                                           optimization_method=optimization_method,
                                                           min_method=min_method, switch_constraints=switch_constraints)
            then = time.time()
            print(f"Data gathering and optimisation took {np.round(then-now, 2)} seconds")
            # Save the computed samples
            store_Abu_data(A, b, u, n_blade_segment, number_samples, va, min_method, file_name_suffix)
            cla_coeffs = coeffs[:degree_cla + 1, 0]
            cda_coeffs = coeffs[degree_cla + 1:, 0]
            print(cla_coeffs)
            print(cda_coeffs)

        if switch_coeffs_grid_plot:
            if switch_recycle_samples:
                A, b, u = retrieve_Abu_data(n_blade_segment, number_samples, va, min_method, file_name_suffix)
            filename_func = lambda ns: f'Saved_figures/{ns}_dp_{n_blade_segment}_bs_{va}_va_{min_method}_{file_name_suffix}.pdf'
            coeffs_grid_row = compute_coeffs_grid_row(A=A, b=b, optimization_method=optimization_method,
                                                      LS_method=LS_method, W_matrix=None, degree_cla=degree_cla,
                                                      degree_cda=degree_cda, min_angle=start_cla_plot,
                                                      max_angle=finish_cla_plot, min_method=min_method,
                                                      switch_constraints=switch_constraints,
                                                      number_samples_lst=number_samples_lst,
                                                      filename_func=filename_func, activate_plotting=True,
                                                      input_storage=u, warm_starts=warm_starts,
                                                      current_coeffs_grid=coeffs_grid,
                                                      warm_start_row_index=i)
            coeffs_grid[[i], :, :] = coeffs_grid_row
            warm_starts = coeffs_grid[i, -1, :]
            data_filename = f'Saved_data/{number_samples_lst[0]}_{number_samples_lst[-1]}_dp_{n_blade_segment_lst[0]}_{n_blade_segment_lst[-1]}_bs_{va}_va_{min_method}_{file_name_suffix}__coeffs_storage.npy'
            with open(data_filename, 'wb') as f:
                np.save(f, coeffs_grid)
    if switch_coeffs_grid_plot:
        plot_coeffs_map(coeffs_grid, degree_cla, degree_cda, number_samples_lst, n_blade_segment_lst)
        figure_filename = f'Saved_figures/{number_samples_lst[0]}_{number_samples_lst[-1]}_dp_{n_blade_segment_lst[0]}_{n_blade_segment_lst[-1]}_bs_{va}_va_{min_method}_{file_name_suffix}.pdf'
        multi_figure_storage(figure_filename, figs=None, dpi=200)

# ----------------------------------------------------------------------------------------------------------------------
# Compute the thrust, thrust moment, x-y in plane force and torque for a short simulation and then plot it
# Local input
if switch_plot_aero:
    n_blade_segment = 100

    # Computations
    n_points = int(total_time/dt + 1)
    F_lst_aero = np.zeros((3, n_points))
    M_lst_aero = np.zeros((3, n_points))
    rotation_angle_lst = np.zeros(n_points)
    rotation_angle = 0
    propeller.set_rotation_angle(rotation_angle)
    for i in range(n_points):
        if not i % 10:
            print(f'Iteration {i} out of {n_points-1}')
        F_damaged, M_damaged = propeller.compute_aero_damaged_FM(n_blade_segment, omega, cla_coeffs, cda_coeffs, body_velocity,
                                                                 pqr, rho)

        F_lst_aero[:, i] += F_damaged.flatten()
        M_lst_aero[:, i] += M_damaged.flatten()
        rotation_angle_lst[i] = rotation_angle
        rotation_angle = propeller.update_rotation_angle(omega, dt)
    plot_FM(np.arange(0, total_time+dt, dt), rotation_angle_lst, F_lst_aero, M_lst_aero, mass_aero="a")

    print(M_damaged, F_damaged)

    F_lst_total = np.dstack((F_lst_mass, F_lst_aero))
    M_lst_total = np.dstack((M_lst_mass, M_lst_aero))
    plot_FM_multiple(np.arange(0, total_time+dt, dt), F_lst_total, M_lst_total, mass_aero="b", x_axis_label="Time [s]")
    for i in plt.get_fignums():
        plt.figure(i)
        plt.legend(["Mass effects", "Aerodynamic effects"], loc=1)

    # with open("Plot_data_storage/F_aero_v0_0_c.npy", 'wb') as f:
    #     np.save(f, F_lst_aero)
    #
    # with open("Plot_data_storage/M_aero_v0_0_c.npy", 'wb') as f:
    #     np.save(f, M_lst_aero)

    F_aero_V0_0 = np.load("Plot_data_storage/F_aero_v0_0_c.npy")
    M_aero_V0_0 = np.load("Plot_data_storage/M_aero_v0_0_c.npy")
    F_lst_total = np.dstack((F_lst_aero, F_aero_V0_0))
    M_lst_total = np.dstack((M_lst_aero, M_aero_V0_0))
    plot_FM_multiple(np.arange(0, total_time+dt, dt), F_lst_total, M_lst_total, mass_aero="a", x_axis_label="Time [s]")
    for i in plt.get_fignums():
        plt.figure(i)
        plt.legend(["Linear $v_i$", "$v_i=0$"], loc=1)

# ----------------------------------------------------------------------------------------------------------------------
# Put all the forces and moments together
# Local input
if switch_plot_mass_aero:
    n_blade_segment = 100

    # Computations
    n_points = int(total_time/dt + 1)
    F_lst = np.zeros((3, n_points))
    M_lst = np.zeros((3, n_points))
    rotation_angle_lst = np.zeros(n_points)
    rotation_angle = 0
    propeller.set_rotation_angle(0)
    for i in range(n_points):
        if not i % 10:
            print(f'Iteration {i} out of {n_points-1}')

        F, M = propeller.compute_mass_aero_FM(n_blade_segment, omega, attitude, cla_coeffs, cda_coeffs, body_velocity,
                                              pqr, rho)

        F_lst[:, i] = F.flatten()
        M_lst[:, i] = M.flatten()
        rotation_angle_lst[i] = rotation_angle
        rotation_angle = propeller.update_rotation_angle(omega, dt)

    # Plot the forces and moments
    plot_FM(np.arange(0, total_time+dt, dt), rotation_angle_lst, F_lst, M_lst, mass_aero='t')

# ----------------------------------------------------------------------------------------------------------------------
# Put all the forces and moments together. The output is the actual thrust and moments generated by the prop
# Local input
if switch_plot_healthy_mass_aero:
    n_blade_segment = 100

    # Computations
    n_points = int(total_time/dt + 1)
    F_healthy_lst = np.zeros((3, n_points))
    M_healthy_lst = np.zeros((3, n_points))
    rotation_angle_lst = np.zeros(n_points)
    rotation_angle = 0
    propeller.set_rotation_angle(0)
    for i in range(n_points):
        if not i % 10:
            print(f'Iteration {i} out of {n_points-1}')

        F, M = propeller.compute_mass_aero_healthy_FM(n_blade_segment, omega, attitude, cla_coeffs, cda_coeffs,
                                                      body_velocity, pqr, rho)

        F_healthy_lst[:, i] = F.flatten()
        M_healthy_lst[:, i] = M.flatten()
        rotation_angle_lst[i] = rotation_angle
        rotation_angle = propeller.update_rotation_angle(omega, dt)

    # Plot the forces and moments
    plot_FM(np.arange(0, total_time+dt, dt), rotation_angle_lst, F_healthy_lst, M_healthy_lst, mass_aero='t')

# ----------------------------------------------------------------------------------------------------------------------
# Obtain the range of values for different degrees of blade damage
if switch_plot_mass_aero_blade_percentage:
    n_blade_segment = 100
    dt = 0.01
    percentage_broken_blade_length_lst = list(range(0, 105, 5))
    F_broken_percentage_max = np.zeros((3, len(percentage_broken_blade_length_lst)))
    F_broken_percentage_min = np.zeros((3, len(percentage_broken_blade_length_lst)))
    M_broken_percentage_max = np.zeros((3, len(percentage_broken_blade_length_lst)))
    M_broken_percentage_min = np.zeros((3, len(percentage_broken_blade_length_lst)))
    broken_percentage_counter = 0
    for percentage_broken_blade_length in percentage_broken_blade_length_lst:
        # Create the propeller and the blades
        bp = [percentage_broken_blade_length, 0, 0]
        propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                              percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                              broken_percentage=bp, plot_chords_twist=False)
        propeller.create_blades()

        # ----------------------------------------------------------------------------------------------------------------------
        # Compute the location of the center of gravity of the propeller and the BladeSection chords
        cg_location = propeller.compute_cg_location()
        average_chords, segment_chords = compute_average_chords(chord_lengths_rt_lst, length_trapezoids_rt_lst, n_blade_segment_lst[0])

        # Computations
        n_points = int(total_time/dt + 1)
        F_lst = np.zeros((3, n_points))
        M_lst = np.zeros((3, n_points))
        rotation_angle_lst = np.zeros(n_points)
        rotation_angle = 0
        propeller.set_rotation_angle(0)
        for i in range(n_points):
            if not i % 10:
                print(f'Blade damage: {percentage_broken_blade_length}. Iteration {i} out of {n_points-1}')
            F, M = propeller.compute_mass_aero_FM(n_blade_segment, omega, attitude, cla_coeffs, cda_coeffs,
                                                  body_velocity, pqr, rho)

            F_lst[:, i] = F.flatten()
            M_lst[:, i] = M.flatten()
            rotation_angle_lst[i] = rotation_angle
            rotation_angle = propeller.update_rotation_angle(omega, dt)

        F_max = np.max(F_lst, axis=1)
        F_min = np.min(F_lst, axis=1)
        M_max = np.max(M_lst, axis=1)
        M_min = np.min(M_lst, axis=1)

        F_broken_percentage_max[:, broken_percentage_counter] = np.array([F_max])
        F_broken_percentage_min[:, broken_percentage_counter] = np.array([F_min])
        M_broken_percentage_max[:, broken_percentage_counter] = np.array([M_max])
        M_broken_percentage_min[:, broken_percentage_counter] = np.array([M_min])
        broken_percentage_counter += 1

    F_broken_percentage_total = np.dstack((F_broken_percentage_max, F_broken_percentage_min))
    M_broken_percentage_total = np.dstack((M_broken_percentage_max, M_broken_percentage_min))
    plot_FM_multiple(percentage_broken_blade_length_lst, F_broken_percentage_total, M_broken_percentage_total,
                     mass_aero="t", x_axis_label="BD [%]")
    for i in plt.get_fignums():
        plt.figure(i)
        plt.legend(["Upper limit", "Lower limit"], loc=2)

    fig = plt.figure(122)
    plt.plot(percentage_broken_blade_length_lst[1:], (F_broken_percentage_min[0, 1:]-F_broken_percentage_min[0, :-1])/5,
             linewidth=4)
    plt.plot(percentage_broken_blade_length_lst[1:], (F_broken_percentage_max[0, 1:]-F_broken_percentage_max[0, :-1])/5,
             linewidth=4)
    plt.xlabel("BD [%]")
    plt.ylabel("$\\frac{d}{d\,BD}\Delta F^B_x$ [N/%]")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
    fig.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)
    plt.grid(True)

    fig = plt.figure(123)
    plt.plot(percentage_broken_blade_length_lst[1:], (M_broken_percentage_min[0,1:]-M_broken_percentage_min[0, :-1])/5,
             linewidth=4)
    plt.plot(percentage_broken_blade_length_lst[1:], (M_broken_percentage_max[0,1:]-M_broken_percentage_max[0 , :-1])/5,
             linewidth=4)
    plt.xlabel("BD [%]")
    plt.ylabel("$\\frac{d}{d\,BD}\Delta M^B_x$ [Nm/%]")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
    fig.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)
    plt.grid(True)

    for i in plt.get_fignums():
        plt.figure(i)
        plt.legend(["Upper limit", "Lower limit"], loc=6)


# M_complete = np.dstack((M_broken_percentage_min, M_broken_percentage_max))
# with open("Plot_data_storage/complete_c.npy", 'wb') as f:
#     np.save(f, M_complete)


# Generation of plots to observe the effect of the chord and the induced velocity
M_complete = np.load("Plot_data_storage/complete_c.npy")
M_S = np.load("Plot_data_storage/no_S_c.npy")   # The blade section area is redefined in BladeSection
M_vi = np.load("Plot_data_storage/no_vi_c.npy")   # The vi is made 0 in Propeller
M_S_vi = np.load("Plot_data_storage/no_S_no_vi_c.npy")  # Both of above

fig = plt.figure(124)

plt.plot(percentage_broken_blade_length_lst[1:], (M_complete[0, 1:, 1] - M_complete[0, :-1, 1]) / 5, linewidth=4, color='#1f77b4', label="Upper limit")
plt.plot(percentage_broken_blade_length_lst[1:], (M_complete[0, 1:, 0] - M_complete[0, :-1, 0]) / 5, linewidth=4, color='#ff7f0e', label="Lower limit")

plt.plot(percentage_broken_blade_length_lst[1:], (M_S[0, 1:, 1] - M_S[0, :-1, 1]) / 5, linewidth=4, color='#2ca02c', label="S=const", linestyle='--', alpha=0.7)
plt.plot(percentage_broken_blade_length_lst[1:], (M_S[0, 1:, 0] - M_S[0, :-1, 0]) / 5, linewidth=4, color='#2ca02c', linestyle='--', alpha=0.7)

plt.plot(percentage_broken_blade_length_lst[1:], (M_vi[0, 1:, 1] - M_vi[0, :-1, 1]) / 5, linewidth=4, color='#d62728', label="$v_i=0$", linestyle='dotted', alpha=0.7)
plt.plot(percentage_broken_blade_length_lst[1:], (M_vi[0, 1:, 0] - M_vi[0, :-1, 0]) / 5, linewidth=4, color='#d62728', linestyle='dotted', alpha=0.7)

plt.plot(percentage_broken_blade_length_lst[1:], (M_S_vi[0, 1:, 1] - M_S_vi[0, :-1, 1]) / 5, linewidth=4, color='#9467bd', label="S=const & $v_i=0$", linestyle='-.', alpha=0.7)
plt.plot(percentage_broken_blade_length_lst[1:], (M_S_vi[0, 1:, 0] - M_S_vi[0, :-1, 0]) / 5, linewidth=4, color='#9467bd', linestyle='-.', alpha=0.7)


plt.xlabel("BD [%]")
plt.ylabel("$\\frac{d}{d\,BD}\Delta M^B_x$ [Nm/%]")
plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
plt.ylim([-0.0005, 0.001])
plt.grid(True)
plt.legend(loc=1)
fig.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)

print("hola")