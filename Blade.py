#!/usr/bin/env python3
"""
Provides the Blade, class for the aerodynamic model identification and computation of forces and moments.

Blade holds all the information related to a single blade and contains a list with all the BladeSection objects that
define the Blade. It is used for calling methods applicable to all the BladeSections which are required for the
computation of the Blade center of gravity, blade area and mass, as well as the moments and forces generated by the
Blade.

Additionally, it computes the contribution for the identification of the lift and drag coefficient relative to a
single blade.
"""

# Modules to import
from math import radians, cos, sin
import numpy as np
from collections import defaultdict

from helper_func import trapezoid_params, compute_chord_blade, compute_average_chord, compute_average_chords, \
    plot_chord_twist, compute_chord_trapezoid
from BladeSection import BladeSection

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"


# Create class Blade
class Blade:
    """
    A class representing a blade and its characteristics.
    """
    def __init__(self, chords, hs, start_twist, final_twist, radius_hub, rotation_direction, initial_angle=0,
                 broken_percentage=0, plot_chords_twist=False):
        """
        Function that computes a blade area and the location of its cg
        :param chords: the base and tip chords of all the trapezoids
        :param hs: the height of all the trapezoids
        :param start_twist: the twist angle at the root
        :param final_twist: the twist angle at the tip
        :param radius_hub: radius of the middle propeller hub
        :param rotation_direction: direction of rotation of the propeller (CW or CCW)
        :param initial_angle: initial angle of the blade with respect to the propeller at the start of the computations
        :param broken_percentage: percentage of the blade that is broken from the tip
        :param plot_chords_twist: whether the twist and the chord along the blade should be plotted
        :return: the total area and the location of the blade cg
        """
        self.healthy_chords = chords
        self.healthy_hs = hs
        self.start_twist = start_twist
        self.healthy_final_twist = final_twist
        self.radius_hub = radius_hub
        self.rotation_direction = rotation_direction
        self.initial_angle = initial_angle
        self.broken_percentage = broken_percentage
        self.plot_chords_twist = plot_chords_twist
        self.blade_area = None       # blade area
        self.healthy_blade_area = 0  # area of complete healthy undamaged blade
        self.y_cg = None             # distance of the cg to the center of rotation
        self.blade_mass = None       # mass of the remaining blade

        self.blade_sections = []          # list to store remaining blade sections
        self.damaged_blade_sections = []  # list to store lost blade sections

        # Compute some parameters in the case that the blade has been damaged
        self.damaged = broken_percentage != 0  # boolean that is True if the blade has been damaged
        self.healthy_h = sum(self.healthy_hs)  # length of the healthy undamaged blade
        self.damaged_h = -1
        if self.damaged:
            # Compute span of remaining blade sections
            self.damaged_h = self.healthy_h * (1 - self.broken_percentage / 100)
            accumulated_healthy_h = np.cumsum(np.array(self.healthy_hs))
            n_complete_trapezoids = np.sum(accumulated_healthy_h < self.damaged_h)
            survived_h = self.healthy_hs[:n_complete_trapezoids]
            self.hs = np.concatenate(
                (self.healthy_hs[:n_complete_trapezoids], np.array([self.damaged_h - sum(survived_h)])))

            # Compute the chord of the remaining blade sections
            new_chords = self.healthy_chords[:n_complete_trapezoids + 1]
            c2 = compute_chord_trapezoid(new_chords[-1], self.healthy_chords[n_complete_trapezoids + 1],
                                         self.healthy_hs[n_complete_trapezoids], 0, self.hs[-1])
            self.chords = np.concatenate((new_chords, np.array([c2])))

            # The twist at the tip of the blade
            self.final_twist = (self.healthy_final_twist -
                                self.start_twist) / self.healthy_h * self.damaged_h + self.start_twist
        else:  # In the case that the blade has not been damaged
            self.hs = self.healthy_hs
            self.chords = self.healthy_chords
            self.final_twist = self.healthy_final_twist

    def compute_blade_params(self):
        """
        Method that computes the location of the center of gravity of the blade and its area
        :return: None
        """
        areas = []
        mass_moment = 0
        h0 = 0
        for i in range(len(self.hs)):
            bc = self.chords[i]
            tc = self.chords[i + 1]
            h = self.hs[i]
            area, y_bar = trapezoid_params(bc, tc, h)
            mass_moment += area * (h0 + y_bar)
            areas.append(area)
            h0 += self.hs[i]
        self.blade_area = sum(areas)
        self.y_cg = 0
        if self.blade_area != 0:
            self.y_cg = mass_moment / self.blade_area

    def compute_blade_mass(self, healthy_mass):
        """
        Method that computes the mass of the blade
        :param healthy_mass: mass of the blade when there is no damage
        :return: The mass of the blade, taking into account any damage. If the blade is damaged, the mass is calculated
        using the healthy mass and the blade's relative area. If not damaged, the mass remains the same as the healthy
        mass.
        """
        if self.damaged:
            self.compute_healthy_blade_area()
            self.blade_mass = healthy_mass * (self.blade_area / self.healthy_blade_area)
        else:
            self.healthy_blade_area = self.blade_area
            self.blade_mass = healthy_mass
        return self.blade_mass

    def compute_healthy_blade_area(self):
        """
        Method that computes the area of the blade when there is no damage
        :return: None
        """
        for i in range(len(self.healthy_hs)):
            bc = self.healthy_chords[i]
            tc = self.healthy_chords[i + 1]
            h = self.healthy_hs[i]
            area, _ = trapezoid_params(bc, tc, h)
            self.healthy_blade_area += area

    def create_blade_sections(self, number_sections):
        """
        Method that creates the BladeSection objects that shape a blade.
        :param number_sections: the number of blade sections in which the blade should be split
        :return: None
        """
        # Compute blade section properties
        blade_length = sum(self.healthy_hs)
        dr = blade_length / number_sections
        average_chords, segment_chords = compute_average_chords(self.healthy_chords, self.healthy_hs, number_sections)

        # Compute the twist for each blade section and each blade section root and tip
        twists_edges = np.linspace(self.start_twist, self.healthy_final_twist, number_sections + 1)
        twist_sections = [(twists_edges[i] + twists_edges[i + 1]) / 2 for i in range(number_sections)]
        if self.plot_chords_twist:
            plot_chord_twist(average_chords, twist_sections)

        # Create blade and append to right list
        for i in range(number_sections):
            blade_section = BladeSection(i, average_chords[i], dr, twist_sections[i], segment_chords[i],
                                         segment_chords[i + 1], self.radius_hub, self.rotation_direction)
            if self.damaged and dr * i >= self.damaged_h:
                self.damaged_blade_sections.append(blade_section)
            else:
                self.blade_sections.append(blade_section)

    def compute_LS_params(self, number_sections, degree_cla, degree_cda, omega, rotation_propeller, propeller_speed,
                          aoa_storage, rho=1.225, inflow_data=None):
        """
        Method that computes a row of the A matrix and b in the Least Squares Ax=b
        :param number_sections: the number of sections in which the blade is divided
        :param degree_cla: the degree of the polynomial that we want to use to approximate the Cl-a curve
        :param degree_cda: the degree of the polynomial that we want to use to approximate the Cd-a curve
        :param omega: the rotational speed of the propeller
        :param rotation_propeller: the angle the propeller has rotated since the start of the computations
        :param propeller_speed: the velocity in 3D space of the complete propeller
        :param aoa_storage: dictionary used to store the aoa experienced by each blade section
        :param rho: the air density
        :param inflow_data: data regarding the inflow field, namely the uniform induced inflow field, induced inflow
        velocity and a lambda function that computes the linear induced field depending on the blade element distance
        from the hub and angle with respect to the inflow.
        :return: contribution of the blade to a single row of the A matrix and a dictionary containing the angles of
        attack experienced by each blade section
        """
        blade_angle = (rotation_propeller + self.initial_angle) % (2 * np.pi)
        if not self.blade_sections and not self.damaged_blade_sections:
            self.create_blade_sections(number_sections)

        # Compute the contribution of each blade section to the A matrix
        LS_terms = np.zeros((2, degree_cla + degree_cda + 2))
        for blade_section in (self.blade_sections + self.damaged_blade_sections):
            V, aoa = blade_section.compute_LS_term_params(omega, blade_angle, propeller_speed, inflow_data)
            aoa_storage[blade_section.section_number].append(aoa[0])

            # Compute contributions of thrust and torque to lift
            for i in range(degree_cla + 1):
                LS_terms[0, i] += blade_section.compute_LS_term_thrust_lift(i, V, aoa)
                LS_terms[1, i] += blade_section.compute_LS_term_torque_lift(i, V, aoa)
            # Compute contributions of thrust and torque to drag
            for j in range(degree_cda + 1):
                LS_terms[0, j + degree_cla + 1] += blade_section.compute_LS_term_thrust_drag(j, V, aoa)
                LS_terms[1, j + degree_cla + 1] += blade_section.compute_LS_term_torque_drag(j, V, aoa)

        LS_terms = LS_terms * 0.5 * rho
        return LS_terms, aoa_storage

    def compute_thrust_moment(self, number_sections, angle_propeller_rotation, omega, propeller_speed, cla_coeffs,
                              cda_coeffs, inflow_data):
        """
        Method that computes the thrust and the moment generated by the thrust force around the propeller hub. This
        is done for the complete blade and the damaged component.
        :param number_sections: number of sections to split the blade
        :param angle_propeller_rotation: the angle the propeller has rotated since the start
        :param omega: the rotation rate of the propeller [rad/s]
        :param propeller_speed: the 3D velocity vector of the propeller system
        :param cla_coeffs: the coefficients used for the computation of the lift coefficient as a function of alpha
        :param cda_coeffs: list of coefficients used for the computation of the cd given the angle of attack
        :param inflow_data:data regarding the inflow field, namely the uniform induced inflow field, induced inflow
        velocity and a lambda function that computes the linear induced field depending on the blade element distance
        from the hub and angle with respect to the inflow.
        :return: the thrust and moments in the x-y plane generated by the remaining and lost blade sections
        """
        blade_angle = (angle_propeller_rotation + self.initial_angle) % (2 * np.pi)
        if not self.blade_sections and not self.damaged_blade_sections:
            self.create_blade_sections(number_sections)

        # Iterate for the healthy/remaining blade sections
        T_remaining = 0
        M_remaining = np.zeros((1, 3))
        for blade_section in self.blade_sections:
            Tr, Mr = blade_section.compute_thrust_moment(omega, propeller_speed, blade_angle, cla_coeffs, cda_coeffs,
                                                         inflow_data)
            T_remaining += Tr
            M_remaining += Mr

        # Iterate for the damaged/lost blade sections
        T_damaged = 0
        M_damaged = np.zeros((1, 3))
        for blade_section in self.damaged_blade_sections:
            Td, Md = blade_section.compute_thrust_moment(omega, propeller_speed, blade_angle, cla_coeffs, cda_coeffs,
                                                         inflow_data)
            T_damaged += Td
            M_damaged += Md

        return T_remaining, T_damaged, M_remaining, M_damaged

    def compute_torque_force(self, number_sections, angle_propeller_rotation, omega, propeller_speed, cla_coeffs,
                             cda_coeffs, inflow_data):
        """
        Method that computes the torque and force in the body x-y plane generated by the healthy part of the blade and
        the damaged sections.
        :param number_sections: number of sections to split the blade
        :param angle_propeller_rotation: the angle the propeller has rotated since the start
        :param omega: the rotation rate of the propeller [rad/s]
        :param propeller_speed: the 3D velocity vector of the propeller system
        :param cla_coeffs: the coefficients used for the computation of the lift coefficient as a function of alpha
        :param cda_coeffs: list of coefficients used for the computation of the cd given the angle of attack
        :param inflow_data:data regarding the inflow field, namely the uniform induced inflow field, induced inflow
        velocity and a lambda function that computes the linear induced field depending on the blade element distance
        from the hub and angle with respect to the inflow.
        :return: the torque and the forces in the body x-y plane
        """
        blade_angle = (angle_propeller_rotation + self.initial_angle) % (2 * np.pi)
        if not self.blade_sections and not self.damaged_blade_sections:
            self.create_blade_sections(number_sections)

        # Iterate for the healthy/remaining blade sections
        Q_remaining = 0
        F_remaining = np.zeros((1, 3))
        for blade_section in self.blade_sections:
            Q_r, F_r = blade_section.compute_torque_force(omega, propeller_speed, blade_angle, cla_coeffs, cda_coeffs,
                                                          inflow_data)
            Q_remaining += Q_r
            F_remaining += F_r

        # Iterate for the damaged/lost blade sections
        Q_damaged = 0
        F_damaged = np.zeros((1, 3))
        for blade_section in self.damaged_blade_sections:
            Q_d, F_d = blade_section.compute_torque_force(omega, propeller_speed, blade_angle, cla_coeffs, cda_coeffs,
                                                          inflow_data)
            Q_damaged += Q_d
            F_damaged += F_d
        return Q_remaining, Q_damaged, F_remaining, F_damaged
