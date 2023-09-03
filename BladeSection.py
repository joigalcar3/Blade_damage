#!/usr/bin/env python3
"""
Provides the BladeSection, class for the aerodynamic model identification and computation of forces and moments.

BladeSection holds all the information related to a single blade element according to BEM theory. It is used for the
computation of the angle of attack and velocity seen by each BladeSection. Additionally, it computes the contribution
of the BladeSection lift and drag to the thrust force and torque.
"""

# Modules to import
import numpy as np
from math import *

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"


class BladeSection:
    """
    Class that stores information about a section within a blade
    """
    def __init__(self, section_number, chord, section_length, average_twist, initial_chord, final_chord, radius_hub,
                 rotation_direction, air_density=1.225):
        """
        Initialize a Blade Section instance.
        :param section_number: the id number of the blade section within its parent Blade class
        :param chord: the average chord of the blade section
        :param section_length: the length of the blade section
        :param average_twist: the average twist experienced along the blade section
        :param initial_chord: the chord at the root of the blade section
        :param final_chord: the chord at the tip of the blade section
        :param radius_hub: the radius of the middle propeller hub
        :param rotation_direction: the direction of rotation of the propelelr
        :param air_density: the density of the air
        :return: None
        """
        self.section_number = section_number
        self.c = chord
        self.dr = section_length
        self.twist = average_twist
        self.c1 = initial_chord
        self.c2 = final_chord
        self.radius_hub = radius_hub
        self.rotation_direction = rotation_direction
        self.rho = air_density

        self.y = (self.section_number + 1/2)*self.dr + self.radius_hub
        self.S = self.c * self.dr
        # Uncomment in order to create S=const curves in Figure 28 of paper: Blade Element Theory Model for UAV Blade
        # Damage Simulation
        # self.S = 9.76*10**(-6)
        self.stall = False  # whether the blade section has stalled due to an extreme angle of attack

    def compute_thrust_moment(self, omega, rotor_speed, position_rotor, cla_coeffs, cda_coeffs, inflow_data):
        """
        Method that computes the thrust produced by the blade section and its corresponding moment about the center of
        the propeller caused by the thrust force
        :param omega: the speed at which the propeller is rotating [rad/s]
        :param rotor_speed: speed at which the drone is flying
        :param position_rotor: position of the propeller coordinate frame relative to the body frame. This information
        is necessary in order to understand how much of the air velocity is perpendicular to the blade [rad]
        :param cla_coeffs: list of coefficients used for the computation of the cl given the angle of attack
        :param cda_coeffs: list of coefficients used for the computation of the cd given the angle of attack
        :param inflow_data:data regarding the inflow field, namely the uniform induced inflow field, induced inflow
        velocity and a lambda function that computes the linear induced field depending on the blade element distance
        from the hub and angle with respect to the inflow.
        :return: blade section thrust and moments in the x-y body plane
        """
        # Compute parameters
        Vl, aoa = self.compute_LS_term_params(omega, position_rotor, rotor_speed, inflow_data)

        # Computation of cl
        cl = 0
        if not self.stall:
            for i in range(len(cla_coeffs)):
                cl += cla_coeffs[i] * aoa**i

        # Computation of cd
        cd = 0
        for i in range(len(cda_coeffs)):
            cd += cda_coeffs[i] * aoa**i

        # Lift equation for the blade section
        dL = 0.5 * self.rho * self.S * Vl ** 2 * cl
        dD = 0.5 * self.rho * self.S * Vl ** 2 * cd
        dT = -(dL * np.cos(self.twist-aoa) - dD * np.sin(self.twist-aoa))  # It is negative because it is in the negative z-direction.

        # Computation of the moment generated by the lift force about the center of the propeller
        r = np.array([[self.y*np.cos(position_rotor), self.y*np.sin(position_rotor), 0]])
        F = np.array([[0, 0, dT.item()]])
        dM = np.cross(r, F)

        return dT, dM

    def compute_torque_force(self, omega, rotor_speed, position_rotor, cla_coeffs, cda_coeffs, inflow_data):
        """
        Method that computes the torque produced by the blade section drag and the corresponding force in the x-y plane
        :param omega: the speed at which the propeller is rotating [rad/s]
        :param rotor_speed: speed at which the drone is flying
        :param position_rotor: position of the propeller coordinate frame relative to the body frame. This information
        is necessary in order to understand how much of the air velocity is perpendicular to the blade [rad]
        :param cla_coeffs: list of coefficients used for the computation of the cl given the angle of attack
        :param cda_coeffs: list of coefficients used for the computation of the cd given the angle of attack
        :param inflow_data:data regarding the inflow field, namely the uniform induced inflow field, induced inflow
        velocity and a lambda function that computes the linear induced field depending on the blade element distance
        from the hub and angle with respect to the inflow.
        :return: blade section torque and forces in the x-y body plane
        """
        # Compute parameters
        Vl, aoa = self.compute_LS_term_params(omega, position_rotor, rotor_speed, inflow_data)

        # Computation of cl
        cl = 0
        if not self.stall:
            for i in range(len(cla_coeffs)):
                cl += cla_coeffs[i] * aoa**i

        # Computation of cd
        cd = 0
        for i in range(len(cda_coeffs)):
            cd += cda_coeffs[i] * aoa**i

        # Drag and lift equations for the blade section
        dL = 0.5 * self.rho * self.S * Vl ** 2 * cl
        dD = 0.5 * self.rho * self.S * Vl ** 2 * cd
        dF_abs = dL * np.sin(self.twist-aoa) + dD * np.cos(self.twist-aoa)

        # Torque equation for the blade section
        dQ = -self.rotation_direction * self.y * dF_abs

        # Computation of the forces in the x-y direction due to the force creating the torque
        angle = position_rotor - self.rotation_direction * np.pi/2
        dF = np.array([[dF_abs.item() * np.cos(angle), dF_abs.item() * np.sin(angle), 0]])

        return dQ, dF

    def compute_aoa(self, rotor_speed, Vx, vi):
        """
        Compute the angle of attack experienced by the blade section
        :param rotor_speed: velocity experienced by the complete motor due to the translation and rotation of the body
        :param Vx: x-component of the velocity experienced by the blade in the blade coordinate frame. This
        means that it is the velocity component perpendicular to the blade
        :param vi: induced velocity
        :return: the blade section angle of attack
        """
        # Computation of the velocity along the z axis and the angle described by the velocity with respect to the x-y
        # coordinate plane
        Vz_bl = -rotor_speed[2, 0] + vi
        velocity_angle = np.arctan(Vz_bl/abs(Vx))

        # Situation when the velocity vector is coming from the back of the blade
        if (-self.rotation_direction*Vx) > 0:
            print(f'{Vx} is greater than 0 for blade section {self.section_number}.')
            aoa = self.twist+np.pi+velocity_angle
            if aoa > np.pi:
                aoa = 2*np.pi-aoa
        else:  # Normal scenario
            aoa = self.twist - velocity_angle

        # When the angle of attack exceeds some limits, it is considered that the blade section has stalled
        if degrees(aoa) < -25 or degrees(aoa) > 25:
            self.stall = True
            print(f'An angle of attack of {degrees(aoa)} degrees means that blade section {self.section_number} '
                  f'has stalled.')
        return aoa

    def compute_velocity(self, omega, position_rotor, rotor_speed, vi):
        """
        Compute the velocity along the chord of the blade section
        :param omega: rotational velocity of the rotor
        :param position_rotor: current rotation of the propeller relative to the body coordinate frame. When the rotor
        position is at an angle of 90 degrees, then the body and the rotor coordinate frames coincide. When the rotor
        angle is 0 degrees, then the x-axis of the propeller is pointing towards the negative body y-axis and the y-axis
        of the propeller is pointing towards the positive body x-axis.
        :param rotor_speed: velocity experienced by the complete motor due to the translation and rotation of the body
        :param vi: induced velocity
        :return: velocity perpendicular to the blade section's chord in the x-y propeller plane (Vl) and the total velocity
        experienced by the blade section along the plane perpendicular to its chord (V_total).
        """
        Vr = self.rotation_direction * omega * self.y   # Velocity at the blade section due to the rotor rotation
        Rot_M = np.array([[np.sin(position_rotor), -np.cos(position_rotor)],
                          [np.cos(position_rotor), np.sin(position_rotor)]])  # Transformation for the velocity vector
        Vxy = rotor_speed[:2]  # x and y components of the body velocity vector of the rotor
        Vxy_bl = np.matmul(Rot_M, Vxy)  # x-y body velocity vector transformed to the blade coordinate frame

        # Once transformed to the blade velocity vector, we are only interested in the x-component, which is
        # perpendicular to the blade. The air velocity experienced by the blade is in the opposite direction of the body
        # movement
        Vx_bl = -Vxy_bl[0, 0]

        # The velocity in the x-direction used for the force computation
        # is the sum of the air velocity due to the body displacement and the propeller rotation
        Vl = Vr + Vx_bl

        # The total velocity experienced by the blade cross section is the sum of the velocity components in the x
        # and z directions
        V_z = -rotor_speed[2] + vi
        V_total = np.sqrt(Vl**2+V_z**2)

        return Vl, V_total

    def compute_LS_term_params(self, omega, position_rotor, rotor_speed, inflow_data):
        """
        Method that computes the velocity and the angle of attack required for the computation of Least Squares
        :param omega: rotational velocity of the rotor
        :param position_rotor: current rotation of the blade relative to the body coordinate frame
        :param rotor_speed: velocity experienced by the complete motor due to the translation and rotation of the body
        :param inflow_data: data regarding the inflow field, namely the uniform induced inflow field, induced inflow
        velocity and a lambda function that computes the linear induced field depending on the blade element distance
        from the hub and angle with respect to the inflow. ["v0", "lambda_0", "induced_velocity_func"(r, psi), "R"]
        :return: the velocity experienced by the blade section used in the lift equation and the angle of attack
        """
        self.stall = False

        # Compute the angle between the speed vector and the blade of the current blade section (azimuth angle: psi)
        Vxy_angle = np.arctan2(rotor_speed[1], rotor_speed[0])
        rotor_speed_vector = np.array([rotor_speed[0, 0], rotor_speed[1, 0]])
        blade_position_vector = np.array([np.cos(position_rotor), np.sin(position_rotor)])
        inter_vector_angle = np.arccos(np.dot(rotor_speed_vector, blade_position_vector) / np.linalg.norm(rotor_speed_vector))
        blade_angle = position_rotor
        if blade_angle > np.pi:  # Keep the angle within [-pi, pi]
            blade_angle = -(2*np.pi-position_rotor)
        if blade_angle < Vxy_angle:    # Transformation from blade angle coordinate frame to azimuth coordinate frame
            inter_vector_angle = -inter_vector_angle
        psi = np.pi + self.rotation_direction * inter_vector_angle

        # Computation of the induced velocity
        vi = inflow_data["induced_velocity_func"](self.y, psi)

        # The following expression is wrong. In all papers r is meant as the distance of the blade element to the rotor
        # hub, instead of the distance to the hub as a percentage of the blade length (as in the equation below)
        # vi = inflow_data["induced_velocity_func"](self.y / inflow_data["R"], psi)  # WRONG

        # Compute the velocities of the vehicle and the angle of attack
        Vx, V_total = self.compute_velocity(omega, position_rotor, rotor_speed, vi)
        aoa = self.compute_aoa(rotor_speed, Vx, vi)
        return V_total, aoa

    def compute_LS_term_thrust_lift(self, term_exponent, V, aoa):
        """
        Method that computes the desired term of the lift equation of the form Vi^2*Si*alpha^term_exponent using the
        lift created by the blade section
        :param term_exponent: the term of the equation that we desire to compute
        :param V: velocity seen by the blade section
        :param aoa: angle of attack seen by the blade section
        :return: contribution of lift to thrust
        """
        dL = self.S * V**2 * aoa**term_exponent
        dT = dL * np.cos(self.twist-aoa)
        if self.stall:
            dT = 0

        return dT

    def compute_LS_term_thrust_drag(self, term_exponent, V, aoa):
        """
        Method that computes the desired term of the lift equation of the form Vi^2*Si*alpha^term_exponent using the
        lift created by the blade section
        :param term_exponent: the term of the equation that we desire to compute
        :param V: velocity seen by the blade section
        :param aoa: angle of attack seen by the blade section
        :return: contribution of drag to thrust
        """
        dD = self.S * V**2 * aoa**term_exponent
        dT = -dD * np.sin(self.twist-aoa)

        return dT

    def compute_LS_term_torque_lift(self, term_exponent, V, aoa):
        """
        Method that computes the desired term of the lift equation of the form Vi^2*Si*alpha^term_exponent using the
        lift created by the blade section
        :param term_exponent: the term of the equation that we desire to compute
        :param V: velocity seen by the blade section
        :param aoa: angle of attack seen by the blade section
        :return: contribution of lift to torque
        """
        dL = self.S * V**2 * aoa**term_exponent
        dQ = -dL * np.sin(self.twist-aoa) * self.y * self.rotation_direction
        if self.stall:
            dQ = 0

        return dQ

    def compute_LS_term_torque_drag(self, term_exponent, V, aoa):
        """
        Method that computes the desired term of the lift equation of the form Vi^2*Si*alpha^term_exponent using the
        lift created by the blade section
        :param term_exponent: the term of the equation that we desire to compute
        :param V: velocity seen by the blade section
        :param aoa: angle of attack seen by the blade section
        :return: contribution of drag to torque
        """
        dD = self.S * V**2 * aoa**term_exponent
        dQ = -dD * np.cos(self.twist-aoa) * self.y * self.rotation_direction

        return dQ
