from pylab import *
import numpy as np
from scipy.interpolate import griddata
from helper_func import *
from user_input import *
from Propeller import Propeller

# User input
max_theta = 2.0 * np.pi
rho = 1.225
R = 0.075
max_r = R
hub_radius = 0.011
BL = R-hub_radius
omega = 1256
A = np.pi*R**2
body_velocity = np.array([[3], [0], [-1]])
pqr = np.array([[0], [0], [0]])
V = np.linalg.norm(body_velocity)
propeller = Propeller(0, n_blades, chord_lengths_rt_lst, length_trapezoids_rt_lst, radius_hub, propeller_mass,
                      percentage_hub_m, state_blades, angle_first_blade, start_twist, finish_twist,
                      broken_percentage=percentage_broken_blade_length, plot_chords_twist=False)
propeller.create_blades()

#Some function to generate values for these points,
#this could be values = np.random.rand(number_points)
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
# induced_velocity_func = lambda r, psi: lambda_0 * (1 + kx * r * np.cos(psi) + ky * r * np.sin(psi)) * omega * R
# induced_velocity_func = lambda r, psi: v0
induced_velocity_func = lambda r, psi: 0

# Create a grid of values, interpolated from our random sample above
psi_lst = np.linspace(0.0, max_theta, 200)
r_lst = np.linspace(0, max_r, 200)
grid_r, grid_theta = np.meshgrid(r_lst, psi_lst)
values = np.zeros(grid_theta.shape)
for i in range(grid_theta.shape[0]):
    for j in range(grid_theta.shape[1]):
        values[i, j] = induced_velocity_func(grid_r[i, j], grid_theta[i, j])

points = np.vstack((grid_r.flatten(), grid_theta.flatten())).T
values = values.flatten()
data = griddata(points, values, (grid_r, grid_theta), method='cubic', fill_value=0)

# Create a polar projection
fig = plt.figure(125)
ax = plt.subplot(projection="polar")
ax.set_rticks([])
# ax.set_xticks([])
im = ax.pcolormesh(psi_lst, r_lst, data.T)
fig.colorbar(im, orientation='vertical', label="$v_i$ [m/s]")
plt.plot(psi_lst, grid_r, color='k', ls='none')
ax.annotate('$V^P_{xy}$', xy=(-0.4, 0.52), xycoords='axes fraction', xytext=(-0.9, 0.5),
            arrowprops=dict(arrowstyle="<|-", color='#1f77b4', lw=4))
ax.annotate('$\\psi=180^\circ$', xy=(-0.35, 0.48), xycoords='axes fraction')
ax.annotate('$0^\circ$', xy=(1.01, 0.48), xycoords='axes fraction')
ax.annotate('$90^\circ$', xy=(0.45, 1.02), xycoords='axes fraction')
ax.annotate('$270^\circ$', xy=(0.42, -0.08), xycoords='axes fraction')
plt.grid(True)
thetagrids((0, 90, 180, 270), labels=("", "", "", ""))
ax.set_rorigin(0)
ax.set_rlim(radius_hub)
# plt.show()
# ax.get_rorigin()
# plt.tight_layout()

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

values = values.flatten()
data = griddata(points, values, (grid_r, grid_theta), method='cubic', fill_value=0)

# Create a polar projection
fig = plt.figure(126)
ax = plt.subplot(projection="polar")
ax.set_rticks([])
# im = ax.pcolormesh(psi_lst, r_lst, data.T, vmin=np.radians(-10), vmax=np.radians(10))
im = ax.pcolormesh(psi_lst, r_lst, data.T, vmin=np.radians(-15), vmax=np.radians(25))
fig.colorbar(im, orientation='vertical', label="$\\alpha$ [rad]")
plt.plot(psi_lst, grid_r, color='k', ls='none')
ax.annotate('$V^P_{xy}$', xy=(-0.4, 0.52), xycoords='axes fraction', xytext=(-0.9, 0.5),
            arrowprops=dict(arrowstyle="<|-", color='#1f77b4', lw=4))
ax.annotate('$\\psi=180^\circ$', xy=(-0.35, 0.48), xycoords='axes fraction')
ax.annotate('$0^\circ$', xy=(1.01, 0.48), xycoords='axes fraction')
ax.annotate('$90^\circ$', xy=(0.45, 1.02), xycoords='axes fraction')
ax.annotate('$270^\circ$', xy=(0.42, -0.08), xycoords='axes fraction')
plt.grid(True)
thetagrids((0, 90, 180, 270), labels=("", "", "", ""))
ax.set_rorigin(0)
ax.set_rlim(radius_hub)
# plt.show()
