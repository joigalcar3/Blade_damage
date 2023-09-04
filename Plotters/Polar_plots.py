#!/usr/bin/env python3
"""
Provides the tool to generate the polar plots that show:
- the effect of drone moving velocity on the angle of attack of the blade (advancing vs retreating blades)
- the effects of the induced velocity (model) on the angle of attack
- the effect of the propeller rotational velocity on the angle of attack

It contains the procedural code to generate the plots.
"""

from pylab import *
import numpy as np
from scipy.interpolate import griddata
from helper_func import *
from user_input import n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass, \
    percentage_hub_m, angle_first_blade, start_twist, finish_twist, percentage_broken_blade_length
from Propeller import Propeller

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

font = {'size': 55,
        'family': "Arial"}
mpl.rc('font', **font)

# User input
max_theta = 2.0 * np.pi  # the maximum value of the polar angle (polar coordinate)
rho = 1.225  # the air density
R = 0.075  # the radius of the propeller
max_r = R  # the maximum value of the radial distance (polar coordinate)
hub_radius = 0.011  # the radius of the hub of the propeller
BL = R - hub_radius  # the blade length
omega = 1256  # the propeller rotational speed
A = np.pi * R ** 2  # the area covered by the rotating propeller
body_velocity = np.array([[3], [0], [-1]])  # the linear velocity of the vehicle
pqr = np.array([[0], [0], [0]])  # the angular velocity of the vehicle
V = np.linalg.norm(body_velocity)  # the magnitude of the linear velocity of the vehicle
type_vi = "L"  # "N"=no vi, "U"=uniform vi, "L"=linear vi

# Obtain propeller object
propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

# Compute the value of the uniform induced velocity with Nelder Mead method
# In theory, the solver does not matter here, but the in-house gradient descent method could be used
T, _ = propeller.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)
angle = np.arcsin(-propeller.propeller_velocity[2] / V)
min_func = lambda x: abs(T - 2 * rho * A * x * np.sqrt((V * np.cos(angle)) ** 2 + (V * np.sin(angle) + x) ** 2))
x0 = np.array([4.5])
bnds = ((0, 20),)
v0 = minimize(min_func, x0, method='Nelder-Mead', tol=1e-6, options={'disp': False}, bounds=bnds).x[0]
lambda_0 = v0 / (omega * R)

# Compute wake skew angle
mu_x = V * np.cos(angle) / (omega * R)
mu_z = V * np.sin(angle) / (omega * R)
Chi = np.arctan(mu_x / (mu_z + lambda_0))

# Compute kx and ky weighting factors
kx = 4.0 / 3.0 * ((1 - np.cos(Chi) - 1.8 * mu_x ** 2) / np.sin(Chi))
ky = -2.0 * mu_x

# Create function for the computation of the linear induced inflow
if type_vi == "L":
    induced_velocity_func = lambda r, psi: lambda_0 * (1 + kx * r * np.cos(psi) + ky * r * np.sin(psi)) * omega * R
elif type_vi == "U":
    induced_velocity_func = lambda r, psi: v0
elif type_vi == "N":
    induced_velocity_func = lambda r, psi: 0
else:
    raise ValueError(f'The selected type of induced velocity, namely {type_vi}, is not correct.')

# Create a grid of values, interpolated from our random sample above and obtain the induced velocity distribution over
# the propeller area
psi_lst = np.linspace(0.0, max_theta, 200)
r_lst = np.linspace(0, max_r, 200)
grid_r, grid_theta = np.meshgrid(r_lst, psi_lst)
values = np.zeros(grid_theta.shape)
for i in range(grid_theta.shape[0]):
    for j in range(grid_theta.shape[1]):
        values[i, j] = induced_velocity_func(grid_r[i, j], grid_theta[i, j])

# Interpolate the values along the surface for smooth plotting
points = np.vstack((grid_r.flatten(), grid_theta.flatten())).T
values = values.flatten()
data = griddata(points, values, (grid_r, grid_theta), method='cubic', fill_value=0)

# Create a polar projection of the induced velocity distribution along the propeller area
fig = plt.figure(1)
ax = plt.subplot(projection="polar")
ax.set_rticks([])
im = ax.pcolormesh(psi_lst, r_lst, data.T)
fig.colorbar(im, orientation='vertical', label="$v_i$, [m/s]", pad=0.065)
plt.plot(psi_lst, grid_r, color='k', ls='none')
ax.annotate('$\overrightarrow{V}^P_{xy}$', xy=(-0.6, 0.52), xycoords='axes fraction', xytext=(-1.03, 0.465),
            arrowprops=dict(arrowstyle="<|-", color='#1f77b4', lw=4))
ax.annotate('$\\psi=180^\circ$', xy=(-0.56, 0.48), xycoords='axes fraction')
ax.annotate('$0^\circ$', xy=(1.01, 0.48), xycoords='axes fraction')
ax.annotate('$90^\circ$', xy=(0.42, 1.02), xycoords='axes fraction')
ax.annotate('$270^\circ$', xy=(0.38, -0.12), xycoords='axes fraction')
plt.grid(True)
thetagrids((0, 90, 180, 270), labels=("", "", "", ""))
ax.set_rorigin(0)
ax.set_rlim(radius_hub)
fig.subplots_adjust(left=0, right=1)

# Obtain the angle of attack at the different polar coordinates of the propeller area for later plotting
values = np.zeros(grid_theta.shape)
for i in range(grid_theta.shape[0]):
    for j in range(grid_theta.shape[1]):
        vi = induced_velocity_func(grid_r[i, j], grid_theta[i, j])

        # Velocity computation
        rotation_direction = propeller.SN[propeller.propeller_number]
        Vr = rotation_direction * omega * grid_r[i, j]  # Velocity at the blade section due to the rotor rotation
        Rot_M = np.array([[np.sin(grid_theta[i, j]), -np.cos(grid_theta[i, j])],
                          [np.cos(grid_theta[i, j]), np.sin(grid_theta[i, j])]])  # Transformation for the velocity vector. This only works because the drone moves forward
        Vxy = propeller.propeller_velocity[:2]  # x and y components of the body velocity vector of the rotor
        Vxy_bl = np.matmul(Rot_M, Vxy)  # x-y body velocity vector transformed to the blade coordinate frame

        # Once transformed to the blade velocity vector, we are only interested in the x-component, which is
        # perpendicular to the blade. The air velocity experienced by the blade is in the opposite direction of the body
        # movement
        Vx_bl = -Vxy_bl[0, 0]

        # The velocity in the x-direction used for the force computation
        # is the sum of the air velocity due to the body displacement and the propeller rotation
        Vl = Vr + Vx_bl

        # Computation of the angle of attack
        Vz_bl = -propeller.propeller_velocity[2, 0] + vi

        velocity_angle = np.arctan(Vz_bl / abs(Vl))
        twist = np.radians(min(27-22/BL*(grid_r[i, j]-radius_hub), 27))
        if (-rotation_direction * Vl) > 0:
            print(f'{Vl} is greater than 0.')
            aoa = twist + np.pi + velocity_angle
            if aoa > np.pi:
                aoa = 2 * np.pi - aoa
        else:
            aoa = twist - velocity_angle
        values[i, j] = aoa

# Interpolate the computed the values along the surface for smooth plotting
values = values.flatten()
data = griddata(points, values, (grid_r, grid_theta), method='cubic', fill_value=0)

# Create a polar projection of the angle of attack over the propeller surface
fig = plt.figure(2)
ax = plt.subplot(projection="polar")
ax.set_rticks([])
im = ax.pcolormesh(psi_lst, r_lst, data.T, vmin=np.radians(-10), vmax=np.radians(10))
# im = ax.pcolormesh(psi_lst, r_lst, data.T, vmin=np.radians(-15), vmax=np.radians(25))  # uncomment for fig 34.c)
cbar = fig.colorbar(im, orientation='vertical', label="$\\alpha$, [rad]", pad=0.065)
cbar.formatter.set_powerlimits((0, 0))
plt.plot(psi_lst, grid_r, color='k', ls='none')
ax.annotate('$\overrightarrow{V}^P_{xy}$', xy=(-0.6, 0.52), xycoords='axes fraction', xytext=(-0.98, 0.465),
            arrowprops=dict(arrowstyle="<|-", color='#1f77b4', lw=4))
ax.annotate('$\\psi=180^\circ$', xy=(-0.56, 0.48), xycoords='axes fraction')
ax.annotate('$0^\circ$', xy=(1.01, 0.48), xycoords='axes fraction')
ax.annotate('$90^\circ$', xy=(0.42, 1.02), xycoords='axes fraction')
ax.annotate('$270^\circ$', xy=(0.38, -0.12), xycoords='axes fraction')
plt.grid(True)
thetagrids((0, 90, 180, 270), labels=("", "", "", ""))
ax.set_rorigin(0)
ax.set_rlim(radius_hub)
fig.subplots_adjust(left=0, right=0.97)
plt.show()
