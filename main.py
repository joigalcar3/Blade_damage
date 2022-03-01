# Code to compute the generated moments and forces due to propeller damage
# Assumptions
# - Homogeneous mass along the blade: the centroid equals the location of the cg
# - The blade is cut parallel to the edge of the propeller such that the remaining polygon is still a trapezoid
# - The Bebop 2 blades are simplified as you trapezoids with parallel sides connected by the long parallel side

# TODO: compute the change of z-force due to the change of weight
# TODO: compute the change of z-force due to the change of lift force
# TODO: compute the change of moment due to the change of the location of the weight
# TODO: compute the change of lift due to the change of the center of pressure
# TODO: compute the change of forces in the x-y plane due to the centrifugal force exerted at the propeller cg.
# TODO: create tests for classes and methods

# Modules to import
from math import radians, cos, sin
from helper_func import trapezoid_params


def blade_params(chords, hs, damaged=False, broken_percentage=0):
    """
    Function that computes a blade area and the location of its cg
    :param broken_percentage: percentage of the blade that is broken from the tip
    :param damaged: whether the blade is damaged
    :param chords: the base and tip chords of all the trapezoids
    :param hs: the height of all the trapezoids
    :return: the total area and the location of the blade cg
    """
    healthy_h = sum(hs)
    damaged_h = -1
    if damaged:
        damaged_h = healthy_h * (1 - broken_percentage / 100)

    areas = []
    mass_moment = 0
    h0 = 0
    for i in range(len(hs)):
        bc = chords[i]
        tc = chords[i + 1]
        h = hs[i]
        if damaged and (h0 + h) > damaged_h:
            tc = bc - (bc - tc) / h * (damaged_h - h0)
            h = damaged_h - h0
        elif damaged and h0 > damaged_h:
            break
        area, y_bar = trapezoid_params(bc, tc, h)
        mass_moment += area * (h0 + y_bar)
        areas.append(area)
        h0 += hs[i]
    total_area = sum(areas)
    y = mass_moment / total_area
    return total_area, y


# User input
percentage_broken_blade_length = 20  # [%]
angle_first_blade = 0  # [deg]
state_blades = [0, 1, 1]

# Propeller info
propeller_mass = 5.17  # [g] measured
percentage_hub_m = 70  # [%]
n_blades = 3  # [-] measured
tip_chord = 0.008  # [m] measured
largest_chord_length = 0.02  # [m] measured
second_segment_length = 0.032  # [m] measured
base_chord = 0.013  # [m] measured
length_blade_origin = 0.076  # [m] measured
radius_hub = 0.012  # [m] measured

chord_lengths_rt_lst = [base_chord, largest_chord_length, tip_chord]
first_segment_length = length_blade_origin - radius_hub - second_segment_length
length_trapezoids_rt_lst = [first_segment_length, second_segment_length]

# Check that the number of states equals the number of blades
if len(state_blades) != n_blades:
    raise Exception("The number of states does not equal the number of blades.")

# Starting computations
hub_m = propeller_mass * (percentage_hub_m / 100)
blade_m = propeller_mass * ((1 - percentage_hub_m) / 100) / n_blades
angle_blades = 360 / n_blades

# Computation location cg of healthy blades
healthy_blade_area, healthy_blade_y = blade_params(chord_lengths_rt_lst, length_trapezoids_rt_lst)

# Computation location cg of damaged blade
damaged_blade_area, damaged_blade_y = blade_params(chord_lengths_rt_lst, length_trapezoids_rt_lst, True,
                                                   percentage_broken_blade_length)
damaged_blade_m = blade_m * (damaged_blade_area / healthy_blade_area)

# Computation of overall cg location
cg_x = 0
cg_y = 0
current_angle = angle_first_blade
for i in range(n_blades):
    if state_blades[i]:
        y_coord = (healthy_blade_y + radius_hub) * cos(radians(current_angle))
        x_coord = (healthy_blade_y + radius_hub) * sin(radians(current_angle))
        mass = blade_m
    else:
        y_coord = (damaged_blade_y + radius_hub) * cos(radians(current_angle))
        x_coord = (damaged_blade_y + radius_hub) * sin(radians(current_angle))
        mass = damaged_blade_m
    cg_x += x_coord * mass / propeller_mass
    cg_y += y_coord * mass / propeller_mass
    current_angle += angle_blades

print(cg_x, cg_y)
