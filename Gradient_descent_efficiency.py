from helper_func import *
from user_input import *
from Propeller import Propeller
import numpy as np
matplotlib.use('TkAgg')

propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, state_blades, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

rho = 1.225
number_samples = 100000

R = sum(propeller.hs) + propeller.radius_hub  # radius of the propeller
A = np.pi * R * R
time_diff_lst = []
scipy_lst = []
grad_lst = []
min_w = -3
n_error_nelder = 0
n_error_grad = 0
for i in range(number_samples):
    body_velocity, pqr, omega = propeller.generate_ls_dp_input(min_w, -min_w, va)

    T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)

    V_inf = np.linalg.norm(propeller.propeller_velocity)
    tpp_V_angle = np.arcsin(-propeller.propeller_velocity[2] / V_inf)

    # Compute time reduction of gradient descend optimisation
    min_func = lambda x: abs(
        T - 2 * rho * A * x[0] * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x[0]) ** 2))
    min_func_2 = lambda x: T - 2 * rho * A * x[0] * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x[0]) ** 2)
    x0 = np.array([4.5])
    bnds = ((0, 20),)
    now_time = time.time()
    v0 = minimize(min_func, x0, method='Nelder-Mead', tol=1e-8, options={'disp': False}, bounds=bnds).x[0]
    then_time = time.time()
    scipy_time = then_time - now_time

    der_opt = lambda x: (-2 * rho * A * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x) ** 2) - \
                         2 * rho * A * x * (V_inf * np.sin(tpp_V_angle) + x) / (
                             np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x) ** 2))) * \
                        min_func_2([x]) / min_func([x])
    x0 = 4.5
    now_time = time.time()
    v0_2 = personal_opt(der_opt, x0, min_func)
    then_time = time.time()
    grad_time = then_time - now_time

    if min_func([v0]) > 1e-5:
        n_error_nelder += 1
    if min_func([v0_2]) > 1e-5:
        n_error_grad += 1

    time_diff = (scipy_time-grad_time)/scipy_time*100
    # print("scipy_time = ", scipy_time)
    # print("grad_time = ", grad_time)
    print(f"Time difference = {time_diff}")
    time_diff_lst.append(time_diff)
    scipy_lst.append(scipy_time)
    grad_lst.append(grad_time)

print(f"Number of Nelder-Mead mistakes = {n_error_nelder}")
print(f"Number of Gradient-Descent mistakes = {n_error_grad}")
print(f"Mean time Nelder-Mead = {np.mean(scipy_lst)}")
print(f"Mean time Gradient-Descent = {np.mean(grad_lst)}")
