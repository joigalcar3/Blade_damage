#!/usr/bin/env python3
"""
Compares the performance between the Nelder-Mead algorithm and the in-house developed Nelder-Mead approach. Please change
the learning rate value in the personal_opt function found within helper_func.py.

It was used to compute the results from Table A.1 in Appendix A. of the author's thesis titled:
"From Data to Prediction: Vision-Based UAV Fault Detection and Diagnosis"
"""

from helper_func import *
from user_input import *
from Propeller import Propeller
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# Create the test propeller
propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

# Constants
rho = 1.225  # air density
number_samples = 10000  # number of optimization iterations

R = sum(propeller.hs) + propeller.radius_hub  # radius of the propeller
A = np.pi * R * R  # area described by the propeller
time_diff_lst = []
scipy_lst = []
grad_lst = []
min_w = -3
max_w = -0.5
n_error_nelder = 0
n_error_grad = 0
for i in range(number_samples):
    if i % 100 == 0:
        print(f'Iteration {i}')
    body_velocity, pqr, omega = propeller.generate_ls_dp_input(min_w, -min_w)
    T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)

    V_inf = np.linalg.norm(propeller.propeller_velocity)
    tpp_V_angle = np.arcsin(-propeller.propeller_velocity[2] / V_inf)

    # Function to minimize
    min_func = lambda x: abs(
        T - 2 * rho * A * x[0] * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x[0]) ** 2))

    # Nelder-Mead optimization
    x0 = np.array([4.5])
    bnds = ((0, 20),)
    now_time = time.time()
    v0 = minimize(min_func, x0, method='Nelder-Mead', tol=1e-8, options={'disp': False}, bounds=bnds).x[0]
    then_time = time.time()
    scipy_time = then_time - now_time

    # In-house developed gradient-descent
    min_func_2 = lambda x: T - 2 * rho * A * x[0] * np.sqrt(
        (V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x[0]) ** 2)
    der_opt = lambda x: (-2 * rho * A * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x) ** 2) - \
                         2 * rho * A * x * (V_inf * np.sin(tpp_V_angle) + x) / (
                             np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x) ** 2))) * \
                        min_func_2([x]) / min_func([x])
    x0 = 4.5
    now_time = time.time()
    v0_2 = personal_opt(der_opt, x0, min_func)
    then_time = time.time()
    grad_time = then_time - now_time

    # If the error is too large, it is deemed as an incorrect sample
    if min_func([v0]) > 1e-5:
        n_error_nelder += 1
    if min_func([v0_2]) > 1e-5:
        n_error_grad += 1

    # Relative difference between the scipy and the gradient times
    time_diff = (scipy_time-grad_time)/scipy_time*100
    print("scipy_time = ", scipy_time)
    print("grad_time = ", grad_time)
    print(f"Time difference = {time_diff}")
    time_diff_lst.append(time_diff)
    scipy_lst.append(scipy_time)
    grad_lst.append(grad_time)

# Print the results
print(f"Number of Nelder-Mead mistakes = {n_error_nelder}")
print(f"Number of Gradient-Descent mistakes = {n_error_grad}")
print(f"Mean time Nelder-Mead = {np.mean(scipy_lst)}")
print(f"Mean time Gradient-Descent = {np.mean(grad_lst)}")
