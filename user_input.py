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

# Matplotlib settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['grid.alpha'] = 0.5

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
percentage_broken_blade_length = 20   # [%]
angle_first_blade = 0                 # [deg] angle of the first blade with respect to the propeller coord. frame
state_blades = [1, 1, 1]              # switches [-]: 1 means that it is healthy
n_blade_segment_lst = list(np.arange(50, 650, 50))               # [-] number of sections in which a single blade is divided
number_samples_lst = list(np.arange(500, 32500, 500))                 # [-] number of data points for the model identification
degree_cla = 2                        # [-] degree of the cl alpha curve polynomial
degree_cda = 2                        # [-] degree of the cd alpha curve polynomial
start_cla_plot = -10                  # [deg] initial alpha value to plot of the cl and cd curves
finish_cla_plot = 30                  # [deg] last alpha value to plot of the cl and cd curves
min_w = -2.5                            # [m/s] minimum vertical velocity considered -2
max_w = -0.5                         # [m/s] maximum vertical velocity considered -0.5
va = 3                                # [m/s] airspeed used for all scenarios
coefficients_identification = True    # Whether the coefficients need to be identified
activate_params_blade_contribution_plotting = False  # Plot that shows how each blade section contributes to the coeffs
LS_method = "GLS"       # the type of least squares used for the identification of the drag and lift coefficients
n_rot_steps = 10        # The number of propeller positions used for taking the average
switch_avg_rot = True   # Switch to activate whether the instantaneous propeller state is used or the rotation average
optimization_method = 'min'   # Whether the opt. method should be Least Squares ("LS") or a scipy.minimization ("min")
min_method = "trust-constr"   # Nonlinear optimization method used to minimize Ax-b: SLSQP, COBYLA
switch_constrains = True    # Whether the optimization should be constrained. Only for COBYLA, SLSQP and trust-constr

# Only for COBYLA, SLSQP and trust-constr accept constraints. Equality constraint means that the constraint function
# result is to be zero whereas inequality means that it is to be non-negative.
# SLSQP: Minimize a scalar function of one or more variables using Sequential Least Squares Programming
# trust-constr: Minimize a scalar function subject to constraints.
# COBYLA: Minimize a scalar function of one or more variables using the Constrained Optimization BY Linear Approximation
# algorithm. It only supports inequality constraints.


# Basic drone state information
body_velocity = np.array([[0, 0, 0]]).T
pqr = np.array([[0, 0, 0]]).T
attitude = np.array([[0, 0, 0]]).T
omega = 1256          # [rad/s]

# Check that the number of states equals the number of blades
if len(state_blades) != n_blades:
    raise Exception("The number of states does not equal the number of blades.")