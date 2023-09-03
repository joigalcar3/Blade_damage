#!/usr/bin/env python3
"""
Provides the Propeller, class for the aerodynamic model identification and computation of forces and moments.

Propeller holds all the information related to the propeller assembly as a whole and contains a list with all the
Blade objects that define the Propeller. It is used for calling methods applicable to all the blades which are required
for the computation of the Propeller center of gravity, as well as the moments and forces generated by the Propeller.

Additionally, it updates the rotation state of the propeller; carries out the identification of the lift and drag
coefficient polynomials by equating the thrust and torque computed with BEM and the Matlab model identified in the
wind tunnel (using an average rotation or a single time instance); computes the uniform inflow field.
"""

# Modules to import
from math import cos, sin, radians, degrees, isclose, pi
import random
from time import time
from collections import defaultdict
from scipy.optimize import minimize

from Blade import Blade
from helper_func import compute_P52, compute_beta, compute_Fn, compute_psi, plot_cla, compute_R_BI, \
    plot_coeffs_params_blade_contribution, iteration_printer, optimize, multi_figure_storage, plot_inputs, personal_opt
from aero_data import *

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"


# Class of propeller that contains blade objects
class Propeller:
    """
    A class representing a propeller and its characteristics.

    Attributes:
        l (float): Distance from the propellers to the body y-axis [m].
        b (float): Distance from the propellers to the body x-axis [m].
        d (np.array): Array representing propeller blade coordinates.
        signr (int): Sign for propeller rotation direction.
        SN (list): List of signs for each blade.
        g (float): Acceleration due to gravity [m/s^2].
    """
    l = 0.0875  # Bebop 2
    b = 0.1150  # Bebop 2
    d = np.array([[l, -b, 0],
                  [l, b, 0],
                  [-l, b, 0],
                  [-l, -b, 0]])
    signr = -1
    SN = [signr, -signr, signr, -signr]
    g = 9.80665

    def __init__(self, propeller_number, number_blades, chords, hs, radius_hub, healthy_propeller_mass,
                 percentage_hub_m, angle_first_blade, start_twist, final_twist, broken_percentage=0,
                 plot_chords_twist=False):
        """
        Initialize a Propeller instance.

        :param propeller_number: Propeller number.
        :param number_blades: Number of blades.
        :param chords: List of chord lengths.
        :param hs: Length of each trapezoid segment.
        :param radius_hub: Radius of the propeller hub.
        :param healthy_propeller_mass: Healthy propeller mass.
        :param percentage_hub_m: Percentage of the total mass that is the hub mass.
        :param angle_first_blade: Angle of the first blade with respect to the propeller coordinate system.
        :param start_twist: Root twist angle.
        :param final_twist: Tip twist angle.
        :param broken_percentage: Percentage of the blade that is broken. Defaults to 0.
        :param plot_chords_twist: Whether to plot chord and twist. Defaults to False.
        :return: None
        """
        self.propeller_number = propeller_number
        self.number_blades = number_blades
        self.chords = chords
        self.hs = hs
        self.radius_hub = radius_hub
        self.healthy_propeller_mass = healthy_propeller_mass
        self.percentage_hub_m = percentage_hub_m
        self.broken_percentage = broken_percentage
        self.angle_first_blade = radians(angle_first_blade)
        self.start_twist = radians(start_twist)
        self.final_twist = radians(final_twist)
        self.plot_chords_twist = plot_chords_twist

        self.blades = []
        self.cg_x = 0
        self.cg_y = 0
        self.cg_r = None

        self.healthy_blade_m = self.healthy_propeller_mass * (1 - self.percentage_hub_m / 100) / self.number_blades
        self.propeller_mass = None

        self.propeller_velocity = None
        self.omega = None
        self.rotation_angle = 0

    def reset_propeller(self, broken_percentage):
        """
        Method to reset the propeller to the default values and change its broken degree.
        :return: None
        """
        self.broken_percentage = broken_percentage
        self.blades = []
        self.cg_x = 0
        self.cg_y = 0
        self.cg_r = None

        self.propeller_mass = None

        self.propeller_velocity = None
        self.omega = None
        self.rotation_angle = 0

    def create_blades(self):
        """
        Function that creates each of the blades objects that are part of a propeller
        :return: None
        """
        current_angle = self.angle_first_blade
        angle_step = 2 * np.pi / self.number_blades  # The angle between blades
        for i in range(self.number_blades):
            if isinstance(self.broken_percentage, list):
                bp = self.broken_percentage[i]
            else:
                bp = self.broken_percentage
            blade = Blade(self.chords, self.hs, self.start_twist, self.final_twist, self.radius_hub,
                          self.SN[self.propeller_number], initial_angle=current_angle,
                          broken_percentage=bp, plot_chords_twist=self.plot_chords_twist)
            self.blades.append(blade)
            current_angle += angle_step

    def compute_blades_params(self):
        """
        Function that computes the location of the cg, the area and the mass of each of the blades, as well as the mass
        of the complete propeller by summing the hub mass with the computed blades' masses.
        :return: None
        """
        if not self.blades:
            self.create_blades()
        blades_mass = 0
        for i in range(self.number_blades):
            blade = self.blades[i]
            blade.compute_blade_params()
            blade_mass = blade.compute_blade_mass(self.healthy_blade_m)
            blades_mass += blade_mass

        # Compute mass of the complete propeller
        self.propeller_mass = blades_mass + self.healthy_propeller_mass * self.percentage_hub_m / 100

    def compute_cg_location(self):
        """
        Function that computes the location of the cg of the complete propeller
        :return: the location of the center of gravity
        """
        if self.propeller_mass is None:
            self.compute_blades_params()

        # Computation of overall cg location
        for i in range(self.number_blades):
            blade = self.blades[i]
            y_coord = (blade.y_cg + self.radius_hub) * sin(blade.initial_angle)
            x_coord = (blade.y_cg + self.radius_hub) * cos(blade.initial_angle)
            self.cg_x += x_coord * blade.blade_mass / self.propeller_mass
            self.cg_y += y_coord * blade.blade_mass / self.propeller_mass
        self.cg_r = np.sqrt(self.cg_x ** 2 + self.cg_y ** 2)

        return [self.cg_x, self.cg_y]

    def compute_cg_forces_moments(self, omega, attitude):
        """
        Method that computes the forces caused by the current location of the center of gravity
        :param omega: propeller rotation
        :param attitude: the attitude of the drone
        :return:
        """
        if self.cg_r is None:
            self.compute_cg_location()

        # Centrifugal force in the propeller coordinate frame
        F_centrifugal = self.propeller_mass * omega ** 2 * self.cg_r
        angle_cg = np.arctan2(self.cg_y, self.cg_x) + self.rotation_angle
        Fx_centrifugal = F_centrifugal * np.cos(angle_cg)
        Fy_centrifugal = F_centrifugal * np.sin(angle_cg)
        F_centrifugal_vector = np.array([[Fx_centrifugal], [Fy_centrifugal], [0]])

        # Moments caused by the shift in cg in the propeller coordinate frame
        M = self.propeller_mass * self.g * self.cg_r
        R_BI = compute_R_BI(attitude)

        # Compute the vector of the gravity force passing through the current center of gravity in the body frame
        Fg_I = np.array([[0], [0], [self.g * self.propeller_mass]])
        Fg_b = np.matmul(R_BI, Fg_I)

        # Compute the vector of the gravity force of the lost blade piece passing through the current center of gravity
        # in the body frame.
        Fg_broken_segment_I = np.array([[0], [0], [-self.g * (self.healthy_propeller_mass - self.propeller_mass)]])
        Fg_broken_segment_b = np.matmul(R_BI, Fg_broken_segment_I)

        # Compute the moment and force vectors that have to be added to the current forces and moments
        r_vector = np.array([[np.cos(angle_cg) * self.cg_r], [np.sin(angle_cg) * self.cg_r], [0]])

        M_vector = np.cross(r_vector.T, Fg_b.T).T
        F_vector = F_centrifugal_vector + Fg_broken_segment_b

        # Perform sanity check with the magnitude of the moment vector
        if all(i == 0 for i in attitude):
            assert isclose(np.linalg.norm(M_vector), M, abs_tol=1e-6), f"The expected moment magnitude is {M}, " \
                                                                       f"whereas the magnitude of the computed vector " \
                                                                       f"is {np.linalg.norm(M_vector)}."

        return F_vector, M_vector

    def compute_lift_torque_matlab(self, body_velocity, pqr, omega, rho=1.225):
        """
        Function that computes the lift of a propeller using the identified polynomials from the gray-box
        aerodynamic model. The equations in this function have been exported from the work of Sihao Sun:
        "Aerodynamic model identification of a quadrotor subjected to rotor failures in the high-speed flight regime."
        :param body_velocity: velocity of the drone in the body reference frame
        :param pqr: rotational velocities of the drone
        :param omega: rotational velocity of the propeller
        :param rho: air density
        :return: thrust and torque
        """
        # Compute the velocity of the propeller due to the linear and angular drone velocities
        self.propeller_velocity = np.cross(pqr.T, self.d[[self.propeller_number], :]).T + body_velocity
        u, v, w = self.propeller_velocity[:].flatten()
        R = sum(self.hs) + self.radius_hub  # radius of the propeller

        # Compute the airspeed experienced by the rotor
        va = np.sqrt(u ** 2 + v ** 2 + w ** 2)

        # Compute the advance ratio
        vv = 0 if (omega * R) == 0 else min(va / (omega * R), 0.6)  # ratio of the airspeed and the tangential velocity

        # Compute the blade angle of attack
        alpha = 0 if np.sqrt(u ** 2 + v ** 2) == 0 else np.arctan(w / np.sqrt(u ** 2 + v ** 2)) * (180 / np.pi)

        # Compute horizontal advance ratio and variable similar to the sideslip angle
        mu = np.sqrt(u ** 2 + v ** 2) / (omega * R)
        lc = w / (omega * R)

        # Compute torque and thrust coefficient corrections from the identified data in the gray-box aerodynamic model
        # research
        if u == 0 and v == 0:
            dCt = 0
            dCq = 0
        else:
            beta = compute_beta(u, v) * (180 / np.pi)
            arm_angle = degrees(np.arctan(self.l / self.b))
            psi_h = compute_psi(beta, arm_angle, self.propeller_number)
            FN_comp = compute_Fn(psi_h, 5, 1, mu, lc)
            dCt = max(min(np.matmul(FN_comp, k_model_2).item(), 0.007), -0.007)

            dCq = max(min(np.matmul(FN_comp, k_model_11).item(), 0.0001), -0.0007)

            vh = np.sqrt(body_velocity[0, 0] ** 2 + body_velocity[1, 0] ** 2)
            dCt = 1 / (1 + np.exp(-6 * (vh - 1))) * dCt
            dCq = 1 / (1 + np.exp(-6 * (vh - 1))) * dCq

        dynhead = rho * omega ** 2 * R ** 2
        area = np.pi * R ** 2  # Area of the circle covered by the rotating propeller

        # Computations for the thrust
        P52_comp = compute_P52(alpha, vv).flatten()  # array of 5th degree polynomial parameters
        Ct = np.dot(P52_comp, k_Ct0.flatten())  # thrust coefficient
        T = (Ct + dCt) * dynhead * area

        # Computations for the torque
        Cq = np.dot(P52_comp, k_Cq0.flatten())
        N = self.SN[self.propeller_number] * (Cq + dCq) * dynhead * area

        return T, N

    def update_rotation_angle(self, omega, delta_t):
        """
        Method that updates the rotation angle of the propeller given its rotational rate and the time that has passed
        since the last update. It is important to keep track of this angle to understand the location of each blade and
        compute their experienced velocity and angle of attack
        :param omega: propeller rotational rate
        :param delta_t: time passed since last update
        :return: current rotation angle of the propeller
        """
        self.omega = omega
        self.rotation_angle += self.omega * delta_t * self.SN[self.propeller_number]
        if self.rotation_angle < 0:
            self.rotation_angle += 2 * np.pi
        self.rotation_angle %= 2 * np.pi
        return self.rotation_angle

    def set_rotation_angle(self, rotation_angle):
        """
        Method to change the rotation angle of the propeller to a given value
        :param rotation_angle: given propeller rotation angle
        :return:
        """
        self.rotation_angle = rotation_angle

    def generate_ls_dp_input(self, min_w, max_w, va=4):
        """
        Method that creates the input velocities for the creation of a data point for the least squares
        :param min_w: the minimum vertical velocity
        :param max_w: the maximum vertical velocity
        :param va: the desired constant airspeed velocity
        :return: the body linear and angular velocities, as well as the propeller rotational velocity
        """
        pqr = np.array([[0], [0], [0]])
        w = random.uniform(min_w, max_w)
        # sign = 1 if random.random() < 0.5 else -1  # uncomment for simulating sampling scheme 1 or 2 from App. C
        # u = sign * np.sqrt(va ** 2 - w ** 2)  # uncomment for simulating sampling scheme 1 from App. C
        # va_local = random.uniform(max(abs(w), 2), va)  # uncomment for simulating sampling scheme 2 from App. C
        # u = sign * np.sqrt(va_local ** 2 - w ** 2)  # uncomment for simulating sampling scheme 2 from App. C
        u = random.uniform(-3, 3)
        body_velocity = np.array([[u], [0], [w]])
        omega = random.uniform(300, 1256+1)  # [rad/s]
        self.rotation_angle = random.uniform(0, 2 * np.pi)

        return body_velocity, pqr, omega

    def compute_cla_coeffs(self, number_samples, number_sections, degree_cla, degree_cda, min_w=-1, max_w=1,
                           va=2, rho=1.225, activate_plotting=True, activate_params_blade_contribution_plotting=False,
                           LS_method="OLS", W_matrix=None, start_plot=-30, finish_plot=30, switch_avg_rot=True,
                           n_rot_steps=10, optimization_method="LS", min_method="Nelder-Mead", switch_constraints=False):
        """
        Main method that computes the cl-alpha coefficients using Least Squares. The average of a complete rotation is
        taken for the coefficients, instead of just one instantaneous propeller rotated position. As seen by many
        papers, we are taking the integral from 0 to 2pi with respect to dpsi. However, here it is done numerically by
        computing the coefficients at some blade rotated positions and later dividing the coefficients by the number of
        rotated positions.
        :param LS_method: Least Squares method used: Ordinary Least Squares (OLS), Weighted LS (WLS), Generalised LS
        (GLS)
        :param W_matrix: the weight matrix used for WLS
        :param finish_plot: last angle of attack to plot
        :param start_plot: first angle of attack to plot
        :param number_samples: the number of samples that will be taken in order to obtain the cla curve
        :param min_w: the minimum vertical velocity
        :param max_w: the maximum vertical velocity
        :param va: the airspeed velocity
        :param rho: the air density
        :param number_sections: number of sections to split the blade
        :param degree_cla: the polynomial degree that we want to use to approximate the Cl-alpha curve
        :param degree_cda: the polynomial degree that we want to use to approximate the Cd-alpha curve
        :param activate_plotting: whether the cl-alpha curve is plotted at the end
        :param activate_params_blade_contribution_plotting: switch for the activation of the plots that show the
        contribution of each blade to the parameters used to identify the lift and drag coefficients
        :param switch_avg_rot: whether the average of a complete rotation needs to be used for the identification or
        only one instance in time
        :param n_rot_steps: number of propeller positions used when taking the average/integral
        :param optimization_method: optimization method used for the computation of the cl and cd coefficients
        :param min_method: optimization method used in scipy.minimization when solving the optimization problem that
        computes the Cd/a and Cl/a curves
        :param switch_constraints: whether constraints should be used in the optimization that computes the Cd/a and
        Cl/acurves
        :return: the dictionary containing all the inputs used to generate the synthetic data from the Matlab
        (gray-box aerodynamic) model, the A matrix, and the b and x vectors
        """
        A = np.zeros((number_samples * 2, degree_cla + degree_cda + 2))  # Coefficient matrix
        b = np.zeros((number_samples * 2, 1))  # Observed data vector
        rot_angles = np.linspace(0, 2 * np.pi, n_rot_steps+1)[:-1]  # Rotation angles considered for average computation
        aoa_storage = defaultdict(list)  # Dictionary for the storage of the angles of attack observed by blade sections
        current_time = time()
        input_storage = defaultdict(list)
        input_storage['title'].append({'body_velocity': "Linear body velocity inputs",
                                       'pqr': "Angular body velocity inputs",
                                       'omega': "Propeller rotational velocity inputs"})
        input_storage['ylabel'].append({'body_velocity': "Linear body velocity [m/s]",
                                        'pqr': "Angular body velocity [rad/s]",
                                        'omega': "Propeller rotational velocity [rad/s]"})
        # Start sample collection
        for i in range(number_samples):
            # Print the time that has passed with respect to the last iteration
            current_time = iteration_printer(i, current_time)

            # Compute the current scenario conditions
            T = -1
            while T < 0:
                # Generate input conditions for gray-box aerodynamic model
                body_velocity, pqr, omega = self.generate_ls_dp_input(min_w, max_w, va)

                # Compute the term corresponding to the b component of LS
                T, N = self.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)

            # Store the information
            input_storage['body_velocity'].append(body_velocity)
            input_storage['pqr'].append(pqr)
            input_storage['omega'].append(omega)
            b[2 * i, 0] = T
            b[2 * i + 1, 0] = N

            # Compute the uniform induced inflow and induced velocity
            inflow_data = self.compute_induced_inflow(T, rho, omega)

            # Compute the terms corresponding to a single row of the A matrix for LS
            LS_terms_blades = [np.zeros((2, degree_cla + degree_cda + 2))] * len(self.blades)
            if not switch_avg_rot:  # In the case that we compute the value at a single rotation and not averaged
                rot_angles = [0]
                n_rot_steps = 1

            # For each rotated angle of the propeller compute the LS terms
            for rot_angle in rot_angles:
                blade_number = 0
                for blade in self.blades:
                    if not switch_avg_rot:
                        rot_angle = self.rotation_angle
                    LS_terms, aoa_storage = blade.compute_LS_params(number_sections, degree_cla, degree_cda, omega,
                                                                    rot_angle, self.propeller_velocity,
                                                                    aoa_storage, inflow_data=inflow_data)
                    # The contribution of each rotation to the A matrix row is already averaged with the number of
                    # rotations considered
                    A[2 * i:2 * i + 2, :] += LS_terms / n_rot_steps
                    LS_terms_blades[blade_number] = LS_terms / n_rot_steps
                    blade_number += 1
            if activate_params_blade_contribution_plotting:
                plot_coeffs_params_blade_contribution(LS_terms_blades, [T, N])

        # Perform Least Squares
        x = optimize(A, b, optimization_method, LS_method=LS_method, W_matrix=W_matrix, degree_cla=degree_cla,
                     degree_cda=degree_cda, min_angle=0, max_angle=finish_plot,
                     min_method=min_method, switch_constraints=switch_constraints, warm_starts=None)

        # Plot the resulting cla and cda curves, the angles of attack seen by each blade section and inputs used to
        # create the thrust and torque dataset from the gray-box aerodynamic model
        if activate_plotting:
            plot_cla(x, A, b, aoa_storage, start_plot, finish_plot, degree_cla, degree_cda)
            plot_inputs(input_storage)
        return x, A, b, input_storage

    def compute_induced_inflow(self, T, rho, omega):
        """
        Method that computes the uniform and linear induced inflow
        :param T: thrust of the propeller, as computed by Matlab
        :param rho: air density
        :param omega: rotational velocity of the propeller
        :return: the uniform and linear induced inflow (lambda_0, induced_velocity_func) and induced velocity (v0)
        """
        R = sum(self.hs) + self.radius_hub  # radius of the propeller
        A = pi * R * R  # area of the propeller
        V_inf = np.linalg.norm(self.propeller_velocity)  # the velocity seen by the propeller
        # V_xy = np.sqrt(
        #     self.propeller_velocity[0] ** 2 + self.propeller_velocity[1] ** 2)  # the velocity projected in the xy plane
        if V_inf != 0:
            tpp_V_angle = np.arcsin(-self.propeller_velocity[2] / V_inf)  # the angle shaped by the tip path plane and the velocity
        else:
            tpp_V_angle = 0

        # Computation of the uniform induced velocity and inflow
        # Function to minimize
        min_func = lambda x: abs(T - 2 * rho * A * x[0] * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 +
                                                                  (V_inf * np.sin(tpp_V_angle) + x[0]) ** 2))

        # Appendix A of the paper: "Blade Element Theory Model for UAV Blade Damage Simulation" compares two approaches
        # for solving this minimization problem, namely the Nelder-Mead optimization or a variant of gradient and
        # descent developed by the author which shows better results in terms of efficiency and performance. Hence, here
        # the in-house developed approach is used.
        # --> 1. Nelder Mead approach for optimization (uncomment the lines below and comment out the gradient descent
        # approach)
        # x0 = np.array([4.5])  # initial condition
        # bnds = ((0, 20),)     # bounds
        # v0 = minimize(min_func, x0, method='Nelder-Mead', tol=1e-6, options={'disp': False}, bounds=bnds).x[0]

        # --> 2. Alternative approach with own gradient descend function. Same result and better time.
        min_func_2 = lambda x: T - 2 * rho * A * x[0] * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 +
                                                                (V_inf * np.sin(tpp_V_angle) + x[0]) ** 2)
        der_func = lambda x: (-2 * rho * A * np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 + (V_inf * np.sin(tpp_V_angle) + x) ** 2) -
                              2 * rho * A * x * (V_inf * np.sin(tpp_V_angle) + x) /
                              (np.sqrt((V_inf * np.cos(tpp_V_angle)) ** 2 +
                                       (V_inf * np.sin(tpp_V_angle) + x) ** 2))) * min_func_2([x]) / min_func([x])
        x0 = np.array([4.5])
        v0 = personal_opt(der_func, x0, min_func)[0]
        lambda_0 = v0 / (omega * R)

        # Decide whether to use the linear or uniform induced inflow models. For instance, if the vehicle is hovering,
        # the uniform inflow model is used
        if V_inf != 0 and np.abs(self.propeller_velocity[0, 0])+np.abs(self.propeller_velocity[1, 0]) != 0:
            # Compute wake skew angle
            mu_x = V_inf * np.cos(tpp_V_angle) / (omega * R)
            mu_z = V_inf * np.sin(tpp_V_angle) / (omega * R)
            Chi = np.arctan(mu_x / (mu_z + lambda_0))

            # Compute kx and ky weighting factors
            kx = 4.0 / 3.0 * ((1 - np.cos(Chi) - 1.8 * mu_x ** 2) / np.sin(Chi))
            ky = -2.0 * mu_x

            # Create function for the computation of the linear induced inflow
            induced_velocity_func = lambda r, psi: lambda_0 * (1 + kx * r * np.cos(psi) + ky * r * np.sin(psi)) * omega * R
        else:
            induced_velocity_func = lambda r, psi: v0
        # induced_velocity_func = lambda r, psi: np.zeros(1)  # uncomment to simulate that induced velocity is zero
        # Dictionary containing the uniform and the linear induced inflow information, namely: the uniform induced
        # inflow (lambda_0), the uniform induced velocity (v0), the induced velocity function and the propeller radius.
        inflow_data = {"lambda_0": lambda_0, "v0": v0, "induced_velocity_func": induced_velocity_func, "R": R}

        return inflow_data

    def compute_thrust_moment(self, number_sections, omega, cla_coeffs, cda_coeffs, inflow_data):
        """
        Method to compute the thrust and the corresponding moment around the propeller hub caused by this forces. This
        is done for the remaining and damaged (flown away) blade sections.
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :param inflow_data: the induced inflow information obtained from the compute_induced_inflow method
        :return: the thrust and the moments in the x and y direction for the healthy and damaged blade sections
        """
        # Create the blades in the case that they were not created
        if not self.blades:
            self.create_blades()

        # Compute the thrust of the propeller
        T_remaining = 0
        T_damaged = 0
        M_remaining = np.zeros((1, 3))
        M_damaged = np.zeros((1, 3))
        for blade in self.blades:
            T_r, T_d, M_r, M_d = blade.compute_thrust_moment(number_sections, self.rotation_angle, omega,
                                                             self.propeller_velocity, cla_coeffs, cda_coeffs,
                                                             inflow_data)
            T_remaining += T_r
            T_damaged += T_d
            M_remaining += M_r
            M_damaged += M_d

        return T_remaining, T_damaged, M_remaining, M_damaged

    def compute_torque_force(self, number_sections, omega, cla_coeffs, cda_coeffs, inflow_data):
        """
        Method to compute the torque generated by the blade, as well as the corresponding force in the x-y plane. This
        is done for the complete propeller and the damaged (flown away) sections.
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :param inflow_data: the induced inflow information obtained from the compute_induced_inflow method
        :return: the torque and the forces in the x and y direction for the healthy and damaged blade sections
        """
        # Create the blades in the case that they were not created
        if not self.blades:
            self.create_blades()

        # Compute the thrust of the propeller
        Q_remaining = 0
        Q_damaged = 0
        F_remaining = np.zeros((1, 3))
        F_damaged = np.zeros((1, 3))
        for blade in self.blades:
            Q_r, Q_d, F_r, F_d = blade.compute_torque_force(number_sections, self.rotation_angle, omega,
                                                            self.propeller_velocity, cla_coeffs, cda_coeffs,
                                                            inflow_data)
            Q_remaining += Q_r
            Q_damaged += Q_d
            F_remaining += F_r
            F_damaged += F_d

        return Q_remaining, Q_damaged, F_remaining, F_damaged

    def compute_aero_damaged_FM(self, number_sections, omega, cla_coeffs, cda_coeffs, body_velocity, pqr, rho=1.225):
        """
        Method to compute the forces and moments caused by the aerodynamic changes
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :param body_velocity: the body 3D linear velocity
        :param pqr: the body 3D angular velocity
        :param rho: the air density
        :return: the aerodynamic forces and moments for the damaged blade sections
        """
        # Create the blades in the case that they were not created
        if not self.blades:
            self.create_blades()

        # Compute the thrust from the Matlab model
        T, _ = self.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)
        inflow_data = self.compute_induced_inflow(T, rho, omega)

        # Obtain the healthy and damaged forces and moments
        _, T_damaged, _, M_damaged = self.compute_thrust_moment(number_sections, omega, cla_coeffs, cda_coeffs,
                                                                inflow_data)
        _, Q_damaged, _, F_damaged = self.compute_torque_force(number_sections, omega, cla_coeffs, cda_coeffs,
                                                               inflow_data)

        F_damaged[0, 2] += T_damaged
        M_damaged[0, 2] += Q_damaged

        return F_damaged, M_damaged

    def compute_aero_healhty_FM(self, number_sections, omega, cla_coeffs, cda_coeffs, body_velocity, pqr, rho=1.225):
        """
        Method to compute the forces and moments caused by the aerodynamics of the remaining blade sections
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :param body_velocity: the body 3D linear velocity
        :param pqr: the body 3D angular velocity
        :param rho: the air density
        :return: the aerodynamic forces and moments for the healthy blade sections
        """
        # Create the blades in the case that they were not created
        if not self.blades:
            self.create_blades()

        # Compute the thrust from the Matlab model
        T, _ = self.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)
        inflow_data = self.compute_induced_inflow(T, rho, omega)

        # Obtain the healthy and damaged forces and moments
        T_healthy, _, M_healthy, _ = self.compute_thrust_moment(number_sections, omega, cla_coeffs, cda_coeffs,
                                                                inflow_data)
        Q_healthy, _, F_healthy, _ = self.compute_torque_force(number_sections, omega, cla_coeffs, cda_coeffs,
                                                               inflow_data)

        F_healthy[0, 2] += T_healthy
        M_healthy[0, 2] += Q_healthy

        return F_healthy, M_healthy

    def compute_mass_aero_FM(self, number_sections, omega, attitude, cla_coeffs, cda_coeffs, body_velocity, pqr,
                             rho=1.225):
        """
        Method to compute the forces and moments caused by the aerodynamic and mass changes
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param attitude: attitude of the drone in the form of Euler angles
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :param body_velocity: the body 3D linear velocity
        :param pqr: the body 3D angular velocity
        :param rho: the air density
        :return: the forces and the moments for the damaged blade sections
        """
        # Computation of forces and moments that derive from the change in mass
        F_cg, M_cg = self.compute_cg_forces_moments(omega, attitude)

        # Computation of moments and forces derived from the loss in an aerodynamic surface
        F_damaged, M_damaged = self.compute_aero_damaged_FM(number_sections, omega, cla_coeffs, cda_coeffs, body_velocity, pqr,
                                                            rho)
        F = F_cg - F_damaged.T
        M = M_cg - M_damaged.T

        return F, M

    def compute_mass_aero_healthy_FM(self, number_sections, omega, attitude, cla_coeffs, cda_coeffs, body_velocity, pqr,
                                     rho=1.225):
        """
        Method to compute the forces and moments caused by the aerodynamics and mass of the remaining blade sections
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param attitude: attitude of the drone in the form of Euler angles
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :param body_velocity: the body 3D linear velocity
        :param pqr: the body 3D angular velocity
        :param rho: the air density
        :return: the forces and the moments for the healthy blade sections
        """
        # Computation of forces and moments that derive from the change in mass
        F_cg, M_cg = self.compute_cg_forces_moments(omega, attitude)

        # Correction of cg forces
        R_BI = compute_R_BI(attitude)
        Fg_healthy_b = np.matmul(R_BI, np.array([[0], [0], [self.g * self.healthy_propeller_mass]]))
        F_cg += Fg_healthy_b

        # Computation of moments and forces derived from the loss in an aerodynamic surface
        F_healthy, M_healthy = self.compute_aero_healhty_FM(number_sections, omega, cla_coeffs, cda_coeffs,
                                                            body_velocity, pqr, rho)
        F = F_cg + F_healthy.T
        M = M_cg + M_healthy.T

        return F, M