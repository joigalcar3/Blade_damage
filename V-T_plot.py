from helper_func import *
from user_input import *
from Propeller import Propeller
from helper_func import multi_figure_storage, plot_coeffs_map, compute_coeffs_grid_row
import matplotlib.pyplot as plt
from itertools import compress
matplotlib.use('TkAgg')

propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, state_blades, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

rho = 1.225
number_samples = 1000

R = sum(propeller.hs) + propeller.radius_hub  # radius of the propeller
A = np.pi * R * R
T_lst = []
V_prod_lst = []
V_lst = []
danger_zone_lst = []
for i in range(number_samples):
    body_velocity, pqr, omega = propeller.generate_ls_dp_input(min_w, max_w, va)

    T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)

    V_inf = np.linalg.norm(propeller.propeller_velocity)
    V_xy = np.sqrt(propeller.propeller_velocity[0] ** 2 + propeller.propeller_velocity[1] ** 2)  # the velocity projected in the xy plane
    tpp_V_angle = np.arccos(V_xy / V_inf)
    if abs(np.sin(tpp_V_angle[0])) >= (2*np.sqrt(2)/3):
        danger_zone = True
    else:
        danger_zone = False

    V_prod = rho*A*V_inf**2*np.sqrt(3)/3

    T_lst.append(T)
    V_prod_lst.append(V_prod)
    V_lst.append(V_inf)
    danger_zone_lst.append(danger_zone)


mask = [V_prod_lst[i]>T_lst[i] for i in range(len(V_prod_lst))]
T_masked = list(compress(T_lst, mask))
V_masked = list(compress(V_lst, mask))
V_prod_masked = list(compress(V_prod_lst, mask))

plt.figure(1)
plt.plot(T_lst, V_prod_lst, "bo")
plt.plot(T_masked, V_prod_masked, "go")
plt.plot(np.arange(0,3,0.1), [i for i in np.arange(0,3,0.1)], "r--")
plt.xlabel("Thrust [N]")
plt.ylabel("$\\rho AV^2\sqrt{3}/3$")
plt.grid(True)

danger_mask = list(compress(mask, danger_zone_lst))
T_double_masked = list(compress(T_lst, danger_mask))
V_doubled_masked = list(compress(V_lst, danger_mask))
plt.figure(2)
plt.plot(T_lst, V_lst, "bo")
plt.plot(T_masked, V_masked, "go")
plt.plot(T_double_masked, V_doubled_masked, "ro", alpha=0.6)
plt.xlabel("Thrust [N]")
plt.ylabel("V")
plt.grid(True)
