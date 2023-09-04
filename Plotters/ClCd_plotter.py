#!/usr/bin/env python3
"""
Provides the tool to generate the cl-alpha and cd-alpha plots.

It contains the procedural code to generate the plots Fig. 15 and 16 of the paper Blade Element Theory Model for UAV
Blade Damage Simulation".
"""

import matplotlib.pyplot as plt
from user_input import cla_coeffs, cda_coeffs
import matplotlib as mpl
import numpy as np

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# Plot formatting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['grid.alpha'] = 0.5
mpl.use('TkAgg')
font = {'size': 42,
        'family': "Arial"}
mpl.rc('font', **font)

# Initialization
red_color_hex = "#d62728"
figure_number = 1

# In-line lambda functions for the lift and drag coefficients given an angle of attack
cl_func = lambda alpha: cla_coeffs[0] + cla_coeffs[1] * alpha + cla_coeffs[2] * alpha**2
cd_func = lambda alpha: cda_coeffs[0] + cda_coeffs[1] * alpha + cda_coeffs[2] * alpha**2
alpha_range = np.radians(np.arange(-10, 30 + 0.001, 0.001))

# Plot the airfoil lift coefficient with respect to the angle of attack
fig = plt.figure(figure_number)
figure_number += 1
plt.plot(np.degrees(alpha_range), list(map(cl_func, alpha_range)), color=red_color_hex, linewidth=2)
plt.xlabel("$\\alpha$ [deg]")
plt.ylabel("$C_l$ [-]")
plt.grid(True)
fig.subplots_adjust(left=0.10, top=0.95, right=0.98, bottom=0.15)
fig.set_size_inches(19.24, 10.55)

# Plot the airfoil drag coefficient with respect to the angle of attack
fig = plt.figure(figure_number)
figure_number += 1
plt.plot(np.degrees(alpha_range), list(map(cd_func, alpha_range)), color=red_color_hex, linewidth=2)
plt.xlabel("$\\alpha$ [deg]")
plt.ylabel("$C_d$ [-]")
plt.grid(True)
fig.subplots_adjust(left=0.10, top=0.95, right=0.98, bottom=0.15)
fig.set_size_inches(19.24, 10.55)
plt.show()
