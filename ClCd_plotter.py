import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from user_input import *

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['grid.alpha'] = 0.5
# mpl.use('Agg')
mpl.use('TkAgg')
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)

red_color_hex = "#d62728"

figure_number = 1

cl_func = lambda alpha: cla_coeffs[0] + cla_coeffs[1] * alpha + cla_coeffs[2] * alpha**2
cd_func = lambda alpha: cda_coeffs[0] + cda_coeffs[1] * alpha + cda_coeffs[2] * alpha**2
alpha_range = np.radians(np.arange(-10, 30 + 0.001, 0.001))


fig = plt.figure(figure_number)
figure_number += 1
plt.plot(np.degrees(alpha_range), list(map(cl_func, alpha_range)), color=red_color_hex, linewidth=2)
plt.xlabel("$\\alpha$ [deg]")
plt.ylabel("$C_l$ [-]")
# plt.xlabel("$\\alpha$, deg")
# plt.ylabel("$C_l$, -")
plt.grid(True)
fig.subplots_adjust(left=0.10, top=0.95, right=0.98, bottom=0.15)
fig.set_size_inches(19.24, 10.55)

fig = plt.figure(figure_number)
figure_number += 1
plt.plot(np.degrees(alpha_range), list(map(cd_func, alpha_range)), color=red_color_hex, linewidth=2)
plt.xlabel("$\\alpha$ [deg]")
plt.ylabel("$C_d$ [-]")
# plt.xlabel("$\\alpha$, deg")
# plt.ylabel("$C_d$, -")
plt.grid(True)
fig.subplots_adjust(left=0.10, top=0.95, right=0.98, bottom=0.15)
fig.set_size_inches(19.24, 10.55)
plt.show()
