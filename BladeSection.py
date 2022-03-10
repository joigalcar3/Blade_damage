import numpy as np
from math import *


class BladeSection:
    """
    Class that stores information about a section within a blade
    """

    def __init__(self, section_number, chord, section_length, average_twist, initial_chord, final_chord, radius_hub,
                 rotation_direction, air_density=1.225):
        self.section_number = section_number
        self.c = chord
        self.dr = section_length
        self.twist = average_twist
        self.c1 = initial_chord
        self.c2 = final_chord
        self.radius_hub = radius_hub
        self.rotation_direction = rotation_direction
        self.rho = air_density

        self.y = (self.section_number+1/2)*self.dr + self.radius_hub
        self.S = self.c * self.dr
        self.stall = False

    def compute_thrust_moment(self, omega, rotor_speed, position_rotor, cla_coeffs, cda_coeffs):
        """
        Method that computes the thrust produced by the blade section and its corresponding moment about the center of
        the propeller caused by the thrust force
        :param omega: the speed at which the propeller is rotating [rad/s]
        :param rotor_speed: speed at which the drone is flying
        :param position_rotor: position of the propeller coordinate frame relative to the body frame. This information
        is necessary in order to understand how much of the air velocity is perpendicular to the blade [rad]
        :param cla_coeffs: list of coefficients used for the computation of the cl given the angle of attack
        :param cda_coeffs: list of coefficients used for the computation of the cd given the angle of attack
        :return:
        """
        # Compute parameters
        Vl, aoa = self.compute_LS_term_params(omega, position_rotor, rotor_speed)

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
        dT = -(dL * np.cos(self.twist-aoa) - dD * np.sin(self.twist-aoa))

        # Computation of the moment generated by the lift force about the center of the propeller
        r = np.array([[self.y*np.cos(position_rotor), self.y*np.sin(position_rotor), 0]])
        F = np.array([[0, 0, dT]])
        dM = np.cross(r, F)

        return dT, dM

    def compute_torque_force(self, omega, rotor_speed, position_rotor, cla_coeffs, cda_coeffs):
        """
        Method that computes the torque produced by the blade section drag and the corresponding force in the x-y plane
        :param omega: the speed at which the propeller is rotating [rad/s]
        :param rotor_speed: speed at which the drone is flying
        :param position_rotor: position of the propeller coordinate frame relative to the body frame. This information
        is necessary in order to understand how much of the air velocity is perpendicular to the blade [rad]
        :param cla_coeffs: list of coefficients used for the computation of the cl given the angle of attack
        :param cda_coeffs: list of coefficients used for the computation of the cd given the angle of attack
        :return:
        """
        # Compute parameters
        Vl, aoa = self.compute_LS_term_params(omega, position_rotor, rotor_speed)

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
        dF = np.array([[dF_abs * np.cos(angle), dF_abs * np.sin(angle), 0]])

        return dQ, dF

    def compute_aoa(self, rotor_speed, Vx):
        """
        Compute the angle of attack
        :param rotor_speed: velocity experienced by the complete motor due to the translation and rotation of the body
        :param Vx: x-component of the velocity experienced by the blade in the blade coordinate frame. This
        means that it is the velocity component perpendicular to the blade
        :return:
        """
        # Computation of the angle of attack
        Vz_bl = -rotor_speed[2, 0]
        velocity_angle = np.arctan(Vz_bl/abs(Vx))

        # Situation when the velocity vector is coming from the back of the blade
        if (-self.rotation_direction*Vx) > 0:
            print(f'{Vx} is greater than 0 for blade section {self.section_number}.')
            aoa = self.twist+np.pi+velocity_angle
            if aoa > np.pi:
                aoa = 2*np.pi-aoa
        else:
            aoa = self.twist - velocity_angle

        if degrees(aoa) < -5 or degrees(aoa) > 25:
            self.stall = True
            print(f'An angle of attack of {degrees(aoa)} degrees means that blade section {self.section_number} '
                  f'has stalled.')
        return aoa

    def compute_velocity(self, omega, position_rotor, rotor_speed):
        """
        Compute the velocity along the chord of the blade section
        :param omega: rotational velocity of the rotor
        :param position_rotor: current rotation of the propeller relative to the body coordinate frame. When the rotor
        position is at an angle of 90 degrees, then the body and the rotor coordinate frames coincide. When the rotor
        angle is 0 degrees, then the x-axis of the propeller is pointing towards the negative body y-axis and the y-axis
        of the propeller is pointing towards the positive body x-axis.
        :param rotor_speed: velocity experienced by the complete motor due to the translation and rotation of the body
        :return:
        """
        Vr = self.rotation_direction * omega * self.y   # Velocity at the blade section due to the rotor rotation
        Rot_M = np.array([[np.sin(position_rotor), -np.cos(position_rotor)],
                          [np.cos(position_rotor), np.sin(position_rotor)]])  # Transformation matrix for the velocity vector
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
        # and y directions
        V_z = rotor_speed[2]
        V_total = np.sqrt(Vl**2+V_z**2)

        return Vl, V_total

    def compute_LS_term_params(self, omega, position_rotor, rotor_speed):
        """
        Method that computes the velocity and the angle of attack required for the computation of Least Squares
        :param omega: rotational velocity of the rotor
        :param position_rotor: current rotation of the propeller relative to the body coordinate frame
        :param rotor_speed: velocity experienced by the complete motor due to the translation and rotation of the body
        :return:
        """
        self.stall = False
        Vx, V_total = self.compute_velocity(omega, position_rotor, rotor_speed)
        aoa = self.compute_aoa(rotor_speed, Vx)
        return V_total, aoa

    def compute_LS_term_thrust_lift(self, term_exponent, V, aoa):
        """
        Method that computes the desired term of the lift equation of the form Vi^2*Si*alpha^term_exponent using the
        lift created by the blade section
        :param term_exponent: the term of the equation that we desire to compute
        :param V: velocity seen by the blade section
        :param aoa: angle of attack seen by the blade section
        :return:
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
        :return:
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
        :return:
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
        :return:
        """
        dD = self.S * V**2 * aoa**term_exponent
        dQ = -dD * np.cos(self.twist-aoa) * self.y * self.rotation_direction

        return dQ
