#!/usr/bin/env python3
"""
Provides the input from the user, as well as information from the propeller.
"""


__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"


# Modules to import
import numpy as np
import matplotlib as mpl
import random
random.seed(1)
np.random.seed(1)

# Matplotlib settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['grid.alpha'] = 0.5
# mpl.use('Agg')
mpl.use('TkAgg')
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)

# Propeller information
propeller_mass_g = 5.07                 # [g] measured 5.07
propeller_mass = propeller_mass_g/1000  # [kg]
blade_mass_g = 1.11                     # [g] measured 1.11. Mass of a single blade
n_blades = 3                            # [-] measured 3
percentage_hub_m = (propeller_mass_g-blade_mass_g*n_blades)/propeller_mass_g*100  # [%] percentage of the total mass
tip_chord = 0.008                 # [m] measured 0.008
largest_chord_length = 0.02       # [m] measured 0.02
second_segment_length = 0.032     # [m] measured 0.032
base_chord = 0.013                # [m] measured 0.013
length_blade_origin = 0.075       # [m] measured 0.076
radius_hub = 0.011                # [m] measured 0.012
start_twist = 27                  # [deg] measured 26.39 [deg]
finish_twist = 5                  # [deg] measured 4.46 [deg]

chord_lengths_rt_lst = [base_chord, largest_chord_length, tip_chord]  # chord lengths at key points (start, max, end)
first_segment_length = length_blade_origin - radius_hub - second_segment_length  # length of the first trapezoid
length_trapezoids_rt_lst = [first_segment_length, second_segment_length]  # list of trapezoid lengths


# User input
percentage_broken_blade_length = [25, 0, 0]   # [%]
angle_first_blade = 0                 # [deg] angle of the first blade with respect to the propeller coord. frame
n_blade_segment_lst = list(np.arange(100, 150, 50))               # [-] number of sections in which a single blade is divided
number_samples_lst = list(np.arange(16000, 17000, 1000))                 # [-] number of data points for the model identification
degree_cla = 2                        # [-] degree of the cl alpha curve polynomial
degree_cda = 2                        # [-] degree of the cd alpha curve polynomial
start_cla_plot = -10                  # [deg] initial alpha value to plot of the cl and cd curves
finish_cla_plot = 30                  # [deg] last alpha value to plot of the cl and cd curves
min_w = -2.5                            # [m/s] minimum vertical velocity considered -2
max_w = -0.5                         # [m/s] maximum vertical velocity considered -0.5
va = 4                                # [m/s] airspeed used for all scenarios
LS_method = "GLS"       # the type of least squares used for the identification of the drag and lift coefficients
n_rot_steps = 10        # The number of propeller positions used for taking the average
optimization_method = 'min'   # Whether the opt. method should be Least Squares ("LS") or a scipy.minimization ("min")
min_method = "SLSQP"   # Nonlinear optimization method used to minimize Ax-b: SLSQP, COBYLA
file_name_suffix = "mod"
coefficients_identification = False    # Whether the coefficients need to be identified
switch_chords_twist_plotting = False  # Whether the chord and twist distribution should be plotted
activate_params_blade_contribution_plotting = False  # Plot that shows how each blade section contributes to the coeffs
switch_recycle_samples = False  # Whether previous samples should be used
switch_avg_rot = True   # Switch to activate whether the instantaneous propeller state is used or the rotation average
switch_constraints = True    # Whether the optimization should be constrained. Only for COBYLA, SLSQP and trust-constr
switch_coeffs_grid_plot = True if len(n_blade_segment_lst)+len(number_samples_lst) > 2 else False  # Whether to plot cl and cd coeffs wrt the #blade sections and #data points
switch_plot_mass = False  # if True, the mass time simulation will be plotted
switch_plot_aero = False  # if True, the aero time simulation will be plotted
switch_plot_mass_aero = False  # if True, the aero and mass time simulation will be plotted
switch_plot_healthy_mass_aero = True  # if True, the aero and mass time simulation will be plotted of the healthy BSs
switch_plot_mass_aero_blade_percentage = False  # if True, the aero and mass maximum and minimum time simulation values will be plotted wrt blade damage

# Only for COBYLA, SLSQP and trust-constr accept constraints. Equality constraint means that the constraint function
# result is to be zero whereas inequality means that it is to be non-negative.
# SLSQP: Minimize a scalar function of one or more variables using Sequential Least Squares Programming
# trust-constr: Minimize a scalar function subject to constraints.
# COBYLA: Minimize a scalar function of one or more variables using the Constrained Optimization BY Linear Approximation
# algorithm. It only supports inequality constraints.


# Basic drone state information
body_velocity = np.array([[2, 0, 0]]).T
pqr = np.array([[0, 0, 0]]).T
attitude = np.array([[0, 0, 0]]).T
omega = 500          # [rad/s]
cla_coeffs = np.array([2.41574347e-01, 5.15236959e+00, -1.22553556e+01])
cda_coeffs = np.array([9.22164567e-03, -7.92542848e-01, 1.51364609e+01])
# cla_coeffs = np.array([2.93049304e-01, 4.47483030e+00, -1.16086661e+01])
# cda_coeffs = np.array([9.33962372e-03, -8.02682724e-01, 1.53301563e+01])
# body_velocity = np.array([[6.30750292e-05, -1.73107276e-06, -6.45597247e-05]]).T
# pqr = np.array([[0, 0, 0]]).T
# attitude = np.array([[0, 0, 0]]).T
# omega = 900       # [rad/s]
rho = 1.225  # [kg/m^3]
total_time = 0.25
dt = 0.001
propeller_number = 1

