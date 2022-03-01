from Blade import Blade
from math import cos, sin, radians, degrees, isclose
import numpy as np
from helper_func import compute_P52, compute_beta, compute_Fn, compute_psi, plot_cla, plot_coeffs_params_blade_contribution
from aero_data import *
import random
from time import time
from collections import defaultdict


# Class of propeller that contains blade objects
class Propeller:
    """
    Method that stores information about the propeller
    """
    l = 0.0875  # distance from the propellers to the body y-axis
    b = 0.1150  # distance from the propellers to the body x-axis
    d = np.array([[l, -b, 0],
                  [l, b, 0],
                  [-l, b, 0],
                  [-l, -b, 0]])
    signr = -1
    SN = [signr, -signr, signr, -signr]
    g = 9.80665

    def __init__(self, propeller_number, number_blades, chords, hs, radius_hub, healthy_propeller_mass,
                 percentage_hub_m, state_blades, angle_first_blade, start_twist, final_twist, broken_percentage=0,
                 plot_chords_twist=False):
        self.propeller_number = propeller_number
        self.number_blades = number_blades
        self.chords = chords
        self.hs = hs
        self.radius_hub = radius_hub
        self.healthy_propeller_mass = healthy_propeller_mass
        self.percentage_hub_m = percentage_hub_m
        self.state_blades = state_blades
        self.broken_percentage = broken_percentage
        self.angle_first_blade = radians(angle_first_blade)
        self.start_twist = radians(start_twist)
        self.final_twist = radians(final_twist)
        self.plot_chords_twist = plot_chords_twist

        self.blades = []
        self.cg_x = 0
        self.cg_y = 0
        self.cg_r = 0

        self.healthy_blade_m = self.healthy_propeller_mass * (1 - self.percentage_hub_m / 100) / self.number_blades
        self.propeller_mass = None

        self.propeller_velocity = None
        self.rotation_angle = 0

    def create_blades(self):
        """
        Function that creates each of the blades objects that are part of a propeller
        :param broken_percentage: the percentage of the blade that is broken. If it is a list, it refers to the broken
        percentage of each of the blades.
        :return:
        """
        current_angle = self.angle_first_blade
        angle_step = 2 * np.pi / self.number_blades
        for i in range(self.number_blades):
            if isinstance(self.broken_percentage, list):
                bp = self.broken_percentage[i]
            else:
                bp = self.broken_percentage
            blade = Blade(self.chords, self.hs, self.start_twist, self.final_twist, self.radius_hub,
                          self.SN[self.propeller_number], initial_angle=current_angle,
                          damaged=not bool(self.state_blades[i]), broken_percentage=bp,
                          plot_chords_twist=self.plot_chords_twist)
            self.blades.append(blade)
            current_angle += angle_step

    def compute_blades_params(self):
        """
        Function that computes the location of the cg, the area and the mass of each of the blades, as well as the mass
        of the complete propeller by summing the hub mass with the computed blades' masses.
        :return:
        """
        if not self.blades:
            self.create_blades()
        blades_mass = 0
        for i in range(self.number_blades):
            blade = self.blades[i]
            blade.compute_blade_params()
            blade_mass = blade.compute_blade_mass(self.healthy_blade_m)
            blades_mass += blade_mass
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
        # Centrifugal force
        F_centrifugal = self.propeller_mass * omega ** 2 * self.cg_r
        angle_cg = np.arctan2(self.cg_y, self.cg_x) + self.rotation_angle
        Fx_centrifugal = F_centrifugal * np.cos(angle_cg)
        Fy_centrifugal = F_centrifugal * np.sin(angle_cg)
        F_centrifugal_vector = np.array([[Fx_centrifugal], [Fy_centrifugal], [0]])

        # Moments caused by the shift in cg
        M = self.propeller_mass * self.g * self.cg_r
        phi = attitude[0, 0]
        theta = attitude[1, 0]
        psi = attitude[2, 0]
        # R_IB = np.array([[np.cos(psi) * np.cos(theta),
        #                   np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi),
        #                   np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
        #                  [np.sin(psi) * np.cos(theta),
        #                   np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi),
        #                   np.sin(psi) * np.sin(theta) * cos(phi) - np.cos(psi) * np.sin(phi)],
        #                  [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]])

        R_BI = np.array([[np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
                         [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                          np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                          np.sin(phi) * np.cos(theta)],
                         [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                          np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                          np.cos(phi) * np.cos(theta)]])

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

        assert isclose(np.linalg.norm(M_vector), M, abs_tol=1e-6), f"The expected moment magnitude is {M}, " \
                                                                   f"whereas the magnitude of the computed vector " \
                                                                   f"is {np.linalg.norm(M_vector)}."

        return F_vector, M_vector

    def compute_lift_torque_matlab(self, body_velocity, pqr, omega, rho=1.225):
        """
        Function that computes the lift of a propeller using the identified polynomials from the aerodynamic model.
        :param body_velocity: velocity of the drone in the body reference frame
        :param pqr: rotational velocities of the drone
        :param omega: rotational velocity of the propeller
        :param rho: air density
        :return:
        """
        self.propeller_velocity = np.cross(pqr.T, self.d[[self.propeller_number], :]).T + body_velocity
        u, v, w = self.propeller_velocity[:].flatten()
        R = sum(self.hs) + self.radius_hub

        va = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        vv = 0 if (omega * R) == 0 else min(va / (omega * R), 0.6)
        alpha = 0 if np.sqrt(u ** 2 + v ** 2) == 0 else np.arctan(w / np.sqrt(u ** 2 + v ** 2)) * (180 / np.pi)

        P52_comp = compute_P52(alpha, vv).flatten()
        Ct = np.dot(P52_comp, k_Ct0.flatten())

        mu = np.sqrt(u ** 2 + v ** 2) / (omega * R)
        lc = w / (omega * R)

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
        area = np.pi * R ** 2
        T = (Ct + dCt) * dynhead * area

        # Computations for the torque
        Cq = np.dot(P52_comp, k_Cq0.flatten())
        N = self.SN[self.propeller_number] * (Cq + dCq) * dynhead * area

        return T, N

    def update_rotation_angle(self, omega, delta_t):
        self.propeller_velocity = omega
        self.rotation_angle += self.propeller_velocity * delta_t * self.SN[self.propeller_number]
        if self.rotation_angle < 0:
            self.rotation_angle += 2 * np.pi
        self.rotation_angle %= 2 * np.pi
        return self.rotation_angle

    def set_rotation_angle(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def generate_ls_dp_input(self, min_w, max_w, va):
        """
        Method that creates the input velocities for the creation of a data point for the least squares
        :param min_w: the minimum vertical velocity
        :param max_w: the maximum vertical velocity
        :param va: the desired constant airspeed velocity
        :return: the body linear and angular velocities, as well as the propeller rotational velocity
        """
        pqr = np.array([[0], [0], [0]])
        w = random.uniform(min_w, max_w)
        sign = 1 if random.random() < 0.5 else -1
        u = sign * np.sqrt(va ** 2 - w ** 2)
        body_velocity = np.array([[u], [0], [w]])
        omega = random.uniform(300, 1256)  # [rad/s]
        self.rotation_angle = random.uniform(0, 2*np.pi)

        return body_velocity, pqr, omega

    def compute_cla_coeffs(self, number_samples, number_sections, degree_cla, degree_cda, min_w=-1, max_w=1, va=2,
                           rho=1.225, activate_plotting=True, activate_params_blade_contribution_plotting=False,
                           LS_method="OLS", W_matrix=0, start_plot=-30, finish_plot=30):
        """
        Function that computes the cl-alpha coefficients using Least Squares
        :param LS_method: Least Squares method used: OLS, WLS, GLS
        :param W_matrix: the matrix used for WLS
        :param finish_plot: last angle of attack to plot
        :param start_plot: first angle of attack to plot
        :param number_samples: the number of samples that will be taken in order to obtain the cla curve
        :param min_w: the minimum vertical velocity
        :param max_w: the maximum vertical velocity
        :param va: the airspeed velocity
        :param rho: the air density
        :param number_sections: number of sections to split the blade
        :param degree_cla: the polynomial degree of the Cl-alpha curve
        :param degree_cda: the degree of the polynomial that we want to use to approximate the Cd-a curve
        :param activate_plotting: whether the cl-alpha curve is plotted at the end
        :return:
        """
        A = np.zeros((number_samples * 2, degree_cla + degree_cda + 2))
        b = np.zeros((number_samples * 2, 1))
        aoa_storage = defaultdict(list)
        current_time = time()
        for i in range(number_samples):
            if i % 5 == 0:
                new_time = time()
                elapsed_time = new_time - current_time
                current_time = new_time
                print(f'Iteration {i}. Elapsed time: {elapsed_time}')

            # Compute the current scenario conditions
            body_velocity, pqr, omega = self.generate_ls_dp_input(min_w, max_w, va)

            # Compute the term corresponding to the b component of LS
            T, N = self.compute_lift_torque_matlab(body_velocity, pqr, omega, rho)
            b[2 * i, 0] = T
            b[2 * i + 1, 0] = N

            # Compute the terms corresponding to the A matrix of LS
            LS_terms_blades = []
            for blade in self.blades:
                LS_terms, aoa_storage = blade.compute_LS_params(number_sections, degree_cla, degree_cda, omega,
                                                                self.rotation_angle, self.propeller_velocity,
                                                                aoa_storage)
                if np.sum(A[2 * i:2 * i + 2, :]) and blade == self.blades[0]:
                    raise Exception("The values should be empty for the coming information")
                A[2 * i:2 * i + 2, :] += LS_terms
                LS_terms_blades.append(LS_terms)
            if activate_params_blade_contribution_plotting:
                plot_coeffs_params_blade_contribution(LS_terms_blades, [T, N])

        # Check what Least Squares method is used    and apply the computation of the unknowns
        if LS_method == "WLS":  # Weighted Least Squares
            if not W_matrix:
                W_straight = np.zeros(A.shape[0])
                W_straight[1::2] = 3600
                W_straight[0::2] = 1
                W_matrix = np.diag(W_straight)
            ATW = np.matmul(A.T, W_matrix)
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(ATW, A)), ATW), b)
        elif LS_method == "GLS":   # Generalized Least Squares
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
            error = b-np.matmul(A, x)
            sigma = np.matmul(error, error.T)
            ATS = np.matmul(A.T, np.linalg.inv(sigma))
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(ATS, A)), ATS), b)
        elif LS_method == "OLS":   # Ordinary Least Squares
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)

        if activate_plotting:
            plot_cla(x, A, b, aoa_storage, start_plot, finish_plot, degree_cla, degree_cda)
        return x, A, b

    def compute_thrust_moment(self, number_sections, omega, propeller_speed, cla_coeffs, cda_coeffs):
        """
        Method to compute the thrust generated by the remaining and damaged (flown away) blade sections
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param propeller_speed: the 3D at which the propeller is moving due to the translation and rotation of the body
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :return:
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
                                                             propeller_speed, cla_coeffs, cda_coeffs)
            T_remaining += T_r
            T_damaged += T_d
            M_remaining += M_r
            M_damaged += M_d

        return T_remaining, T_damaged, M_remaining, M_damaged

    def compute_torque_force(self, number_sections, omega, propeller_speed, cla_coeffs, cda_coeffs):
        """
        Method to compute the thrust generated by the remaining and damaged (flown away) blade sections
        :param number_sections: total number of sections in which a healthy blade is split up
        :param omega: the rotation rate of the propeller
        :param propeller_speed: the 3D at which the propeller is moving due to the translation and rotation of the body
        :param cla_coeffs: the coefficients that relate the angle of attack to the airfoil lift coefficient
        :param cda_coeffs: the coefficients that relate the angle of attack to the airfoil drag coefficient
        :return:
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
                                                            propeller_speed, cla_coeffs, cda_coeffs)
            Q_remaining += Q_r
            Q_damaged += Q_d
            F_remaining += F_r
            F_damaged += F_d

        return Q_remaining, Q_damaged, F_remaining, F_damaged
