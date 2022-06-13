from helper_func import *
from user_input import *
from Propeller import Propeller
from helper_func import multi_figure_storage, plot_coeffs_map, compute_coeffs_grid_row
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
from itertools import compress
matplotlib.use('TkAgg')

propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

rho = 1.225
number_samples = 100000

R = sum(propeller.hs) + propeller.radius_hub  # radius of the propeller
A = np.pi * R * R
T_lst = []
V_prod_lst = []
V_lst = []
danger_zone_lst = []
min_w = -3
for i in range(number_samples):
    body_velocity, pqr, omega = propeller.generate_ls_dp_input(min_w, -min_w, va)

    T, N = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)

    V_inf = np.linalg.norm(propeller.propeller_velocity)
    V_xy = np.sqrt(propeller.propeller_velocity[0] ** 2 + propeller.propeller_velocity[1] ** 2)  # the velocity projected in the xy plane
    tpp_V_angle = np.arcsin(-propeller.propeller_velocity[2] / V_inf)
    if abs(np.sin(tpp_V_angle[0])) >= (2*np.sqrt(2)/3) and np.sin(tpp_V_angle[0]) < 0:
        danger_zone = True
    else:
        danger_zone = False

    V_prod = rho*A*V_inf**2*np.sqrt(3)/3

    T_lst.append(T)
    V_prod_lst.append(V_prod)
    V_lst.append(V_inf)
    danger_zone_lst.append(danger_zone)

mask = [V_prod_lst[i] > T_lst[i] for i in range(len(V_prod_lst))]
T_masked = list(compress(T_lst, mask))
V_masked = list(compress(V_lst, mask))
V_prod_masked = list(compress(V_prod_lst, mask))

plt.figure(1)
plt.scatter(T_lst, V_prod_lst, [0.1]*len(T_lst), color="blue", label='T>$\\rho AV^2\sqrt{3}/3$')
plt.scatter(T_masked, V_prod_masked, [0.2]*len(T_masked), color="green", label='T<$\\rho AV^2\sqrt{3}/3$')
plt.plot(np.arange(0, 3, 0.1), [i for i in np.arange(0, 3, 0.1)], "r--", label='T=$\\rho AV^2\sqrt{3}/3$')
plt.xlabel("Thrust [N]")
plt.ylabel("$\\rho AV^2\sqrt{3}/3$")
plt.ylim([0, max(V_prod_lst)+0.1])
plt.grid(True)
lgnd = plt.legend()
for legendHandle in lgnd.legendHandles:
    legendHandle._sizes = [30]

danger_mask = np.array(mask) * np.array(danger_zone_lst)
T_danger = np.array(T_lst)[danger_zone_lst]
V_danger = np.array(V_lst)[danger_zone_lst]
T_double_masked = np.array(T_lst)[danger_mask]
V_doubled_masked = np.array(V_lst)[danger_mask]
points = np.hstack((np.reshape(V_danger, [-1, 1]), np.reshape(T_danger, [-1, 1])))
hull = ConvexHull(points)

plt.figure(2)
plt.scatter(V_lst, T_lst, [0.1]*len(T_lst), marker="o", color="blue", alpha=0.5, label='T>$\\rho AV^2\sqrt{3}/3$ & $\\sin{\\alpha_d} \\geq -\\frac{2\\sqrt{2}}{2}$')
plt.scatter(V_masked, T_masked, [0.8]*len(T_masked), marker="o", color="green", alpha=0.5, label='T<$\\rho AV^2\sqrt{3}/3$')
plt.scatter(V_danger, T_danger, [0.8]*len(T_danger), marker="o", color="magenta", alpha=0.5, label='$\\sin{\\alpha_d} \\leq -\\frac{2\\sqrt{2}}{2}$')
if V_doubled_masked:
    plt.scatter(V_doubled_masked, T_double_masked, [2]*len(T_double_masked), color="red", alpha=0.5, label='T<$\\rho AV^2\sqrt{3}/3$ & $\\sin{\\alpha_d} \\leq -\\frac{2\\sqrt{2}}{2}$')
V_range = np.arange(min(V_lst), max(V_lst) + 0.2, 0.01)
plt.plot(V_range, [rho*A*V**2*np.sqrt(3)/3 for V in V_range], "k--", label='T=$\\rho AV^2\sqrt{3}/3$')
plt.plot(points[hull.simplices[0], 0], points[hull.simplices[0], 1], 'k-', label="Convex hull: $\\sin{\\alpha_d} \\leq -\\frac{2\\sqrt{2}}{2}$")
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.ylim([min(T_lst)-0.2, max(T_lst)+0.2])
plt.xlim([min(V_lst)-0.2, max(V_lst)+0.5])
plt.xlabel("Velocity [m/s]")
plt.ylabel("Thrust [N]")
plt.grid(True)
lgnd = plt.legend(loc="upper right")
for legendHandle in lgnd.legendHandles:
    legendHandle._sizes = [30]

print("hola")
