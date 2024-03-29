#!/usr/bin/env python3
"""
Provides the helper functions, workhorse of the whole blade damage implementation

It contains functions that carry out simple mathematical/geometrical computations, implements the aerodynamic matlab
model, implements the model identification and all the plotters.
"""

# Modules to import
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import radians, degrees
import time
import os
import pickle
import inspect
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, ScalarFormatter, IndexLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.stattools import durbin_watson
import matplotlib

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# General global params
start_time = time.time()
figure_number = 1


# Helper functions related to mathematical/geometric operations
# %%
def trapezoid_params(bc, tc, h):
    """
    Function that computes the area and location of the trapezoid cg from the bc
    :param bc: base chord, closest to the hub
    :param tc: tip chord
    :param h: length of trapezoid
    :return: the area and the location of the cg of the trapezoid
    """
    area = compute_trapezoid_area(bc, tc, h)
    y_bar = compute_trapezoid_cg(bc, tc, h)
    return area, y_bar


def compute_trapezoid_area(bc, tc, h):
    """
    Function that computes the area of a trapezoid given its dimensions
    :param bc: the base chord
    :param tc: the tip chord
    :param h: the height of the trapezoid
    :return: the area of the trapezoid
    """
    area = (tc + bc) * h / 2
    return area


def compute_trapezoid_cg(bc, tc, h):
    """
    Compute the location of the trapezoid center of gravity assuming that the material density is constant
    :param bc: the base chord
    :param tc: the tip chord
    :param h: the trapezoid length
    :return: the location along the h direction where the center of gravity is located
    """
    y_bar = (2 * tc + bc) / (tc + bc) * h / 3
    return y_bar


def compute_chord_blade(chords, hs, pos):
    """
    Given a location along the blade, compute the chord
    :param chords: list of all blade chords
    :param hs: list of all blade segment distances
    :param pos: desired distance from the root
    :return: chord at a certain location along the blade
    """
    h0 = 0
    for i in range(len(hs)):
        h = hs[i]
        if (h0 + h) > pos:
            bc = chords[i]
            tc = chords[i + 1]
            chord = compute_chord_trapezoid(bc, tc, h, h0, pos)
            return chord
        h0 += h
    raise Exception("The provided blade position does not match the blade size.")


def compute_chord_trapezoid(bc, tc, h, h0, pos):
    """
    Function that computes the chord of a trapezoid at a specific location along the h direction
    :param bc: chord at the base
    :param tc: chord at the tip
    :param h: length of the trapezoid
    :param h0: position along the blade at which the trapezoid base is located
    :param pos: position along the blade at which we would like to compute the chord
    :return: chord length
    """
    chord = bc - (bc - tc) / h * (pos - h0)
    return chord


def compute_average_chord(chords, hs, pos_start, pos_end):
    """
    Function to compute the average chord of a blade section
    :param chords: chord lengths of the trapezoids that together make the blade
    :param hs: list of all the trapezoid lengths that together make the blade
    :param pos_start: start location of the blade section
    :param pos_end: end location of the blade section
    :return: average chord length of a blade section
    """
    h0 = 0
    area = 0
    distance = pos_end - pos_start
    for i in range(len(hs)):
        h = hs[i]
        bc = chords[i]
        tc = chords[i + 1]
        if (h0 + h) > pos_start >= h0:
            # Chord at the root of the blade section
            c1 = compute_chord_trapezoid(bc, tc, h, h0, pos_start)
            if (h0 + h) <= pos_end:  # in the case that the end location of the blade section is in another trapezoid
                area += compute_trapezoid_area(c1, tc, h - pos_start)
                c1 = tc
                pos_start = h0 + h

            if (h0 + h) > pos_end >= h0:  # the end location of the blade section is in the current trapezoid
                # Chord at the tip of the blade section
                c2 = compute_chord_trapezoid(bc, tc, h, h0, pos_end)
                area += compute_trapezoid_area(c1, c2, pos_end - pos_start)
        h0 += h

    average_chord = area / distance
    return average_chord


def compute_average_chords(chords, hs, n_segments):
    """
    Function that computes the average chords of all the blade section
    :param chords: list with all the trapezoid sections' chords
    :param hs: list with all the trapezoid sections' lengths
    :param n_segments: number of blade sections
    :return: average chords of all the blade sections and the chords at the root and tip of the blade sections
    """

    def update_chords_h(counter, h_origin):
        """
        Function that retrieves the information related to a blade section
        :param counter: current trapezoid index
        :param h_origin: start location of the blade section along the blade axis
        :return: the length of the current blade trapezoid, its root and tip chords, the trapezoid counter and the
        location along the blade where the trapezoid root is located.
        """
        if counter >= 0:
            h_origin += hs[counter]
        counter += 1
        current_h = hs[counter]
        current_bc = chords[counter]
        current_tc = chords[counter + 1]
        return current_h, current_bc, current_tc, counter, h_origin

    length_blade = sum(hs)  # length of the blade
    length_segment = length_blade / n_segments  # length of each blade section

    segment = 0
    h, bc, tc, trapezoid_count, h0 = update_chords_h(-1, 0)
    average_chords = []
    segment_chords = [bc]
    for i in range(n_segments):
        area_1 = 0
        pos_c1 = segment * length_segment  # location of the blade section's root chord along the blade
        pos_c2 = (segment + 1) * length_segment  # location of the blade section's tip chord along the blade
        c1 = segment_chords[segment]  # chord of the blade section's root
        if pos_c2 > (h0 + hs[trapezoid_count]):  # In the case that the blade section's tip is in the next trapezoid
            # Compute the area of the part of the blade section in the first trapezoid
            area_1 = compute_trapezoid_area(c1, tc, h0 + h - pos_c1)
            # Retrieve the information from the second trapezoid
            h, bc, tc, trapezoid_count, h0 = update_chords_h(trapezoid_count, h0)
        # Compute the location of the blade section's tip chord given its location and the information of the trapezoid
        # where it is located
        c2 = compute_chord_trapezoid(bc, tc, h, h0, pos_c2)

        if not bool(area_1):  # when the blade section is in a single trapezoid
            area = compute_trapezoid_area(c1, c2, length_segment)
        else:  # when it is found between two trapezoids
            area = area_1 + compute_trapezoid_area(bc, c2, pos_c2 - h0)
        average_chord = area / length_segment  # compute the average chord of the blade section
        average_chords.append(average_chord)
        segment_chords.append(c2)
        segment += 1
    return average_chords, segment_chords


def compute_R_BI(attitude):
    """
    Compute transformation matrix from the inertial to the body coordinate frames
    :param attitude: the attitude of the vehicle
    :return: R_BI (transformation from inertial to body coordinate frame)
    """
    phi = attitude[0, 0]
    theta = attitude[1, 0]
    psi = attitude[2, 0]

    R_BI = np.array([[np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
                     [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                      np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                      np.sin(phi) * np.cos(theta)],
                     [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                      np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                      np.cos(phi) * np.cos(theta)]])  # Transformation from the inertial to the body coordinate frame
    return R_BI


# Helper functions related to aerodynamic Matlab model (gray-box aerodynamic model)
# %%
def compute_P52(x1, x2):
    """
    Function from the Matlab Bebop 2 model for computing a 5th degree polynomial with two variables
    :param x1: first variable
    :param x2: second variable
    :return: array of the polynomial terms elevated to the corresponding exponents
    """
    U = 1
    A1 = 1 * U
    A2 = x1 * U
    A3 = x2 * U
    A4 = x2 ** 2 * U
    A5 = x1 * x2 * U
    A6 = x2 ** 2 * U
    A7 = x1 ** 3 * U
    A8 = x1 ** 2 * x2 * U
    A9 = x1 * x2 ** 2 * U
    A10 = x2 ** 3 * U
    A11 = x1 ** 4 * U
    A12 = x1 ** 3 * x2 * U
    A13 = x1 ** 2 * x2 ** 2 * U
    A14 = x1 * x2 ** 3 * U
    A15 = x2 ** 4 * U
    A16 = x1 ** 5 * U
    A17 = x1 ** 4 * x2 * U
    A18 = x1 ** 3 * x2 ** 2 * U
    A19 = x1 ** 2 * x2 ** 3 * U
    A20 = x1 * x2 ** 4 * U
    A21 = x2 ** 5 * U

    A_p52 = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,
                       A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21]])
    return A_p52


def compute_beta(u, v):
    """
    Method that computes the sideslip angle given the body coordinate frame
    :param u: velocity along the x-axis
    :param v: velocity along the y-axis
    :return: sideslip angle
    """
    beta = abs(np.arctan(v / u))

    if u < 0 and v < 0:
        beta = np.pi - beta
    elif u < 0 and v > 0:
        beta += np.pi
    elif u > 0 and v > 0:
        beta = 2 * np.pi - beta
    elif u == 0 and v == 0:
        beta = 0

    return beta


def compute_psi(beta, arm_angle, propeller_number):
    """
    Function to compute the psi value from the aerodynamic Matlab model (gray-box aerodynamic model)
    :param beta: sideslip angle
    :param arm_angle: angle of the rotor arm with the body
    :param propeller_number: number of the propeller
    :return: value of psi parameter
    """
    psi_h = 0
    if propeller_number == 0:
        psi_h = beta - (360 + 90 - arm_angle)
    elif propeller_number == 1:
        psi_h = beta - (360 - 90 + arm_angle)
    elif propeller_number == 2:
        psi_h = beta - (180 + 90 - arm_angle)
    elif propeller_number == 3:
        psi_h = beta - (180 - 90 + arm_angle)

    psi_h /= (180 / np.pi)
    psi_h %= (2 * np.pi)

    if propeller_number == 1 or propeller_number == 3:
        psi_h = 2 * np.pi - psi_h

    return psi_h


def compute_P32(x1, x2, U):
    """
    Function obtained from the Matlab Bebop 2 model (gray-box aerodynamic model) to compute a 3rd order polynomial with
    two variables
    :param x1: first variable
    :param x2: second variable
    :param U: additional input
    :return: 3rd order polynomial with two variables parameter array
    """
    A2 = x1 * U
    A3 = x2 * U
    A4 = x1 ** 2 * U
    A5 = x2 ** 2 * U
    A6 = x1 * x2 * U
    A7 = x1 ** 3 * U
    A8 = x2 ** 3 * U
    A9 = x1 * x2 ** 2 * U
    A10 = x1 ** 2 * x2 * U
    A_p33 = [A2, A3, A4, A5, A6, A7, A8, A9, A10]

    return A_p33


def compute_Fn(x, n, U, alpha, beta):
    """
    Function retrieved from the Matlab Bebop 2 model (gray-box aerodynamic mdoel) that computes the nth order Fourier
    series
    :param x: variable for obtaining the Fourier series
    :param n: degree of the Fourier series
    :param U: additional input
    :param alpha: horizontal advance ratio
    :param beta: sideslip angle equation
    :return: parameter AF that will be used for the thrust and torque coefficient correction
    """
    AF = []
    for i in range(1, n + 1):
        AF.append(compute_P32(alpha, beta, np.sin(i * x) * U))
        AF.append(compute_P32(alpha, beta, np.cos(i * x) * U))
    AF = np.reshape(np.array(AF).flatten(), [1, -1])
    return AF


def rpm_rads(rpm):
    """
    Function to transform revolutions per minute to rad/s
    :param rpm: revolutions per minute
    :return: rotational speed in rad/s
    """
    rads = rpm * 2 * np.pi / 60
    return rads


def iteration_printer(i, current_time):
    """
    Prints the current iteration and the time that has passed between iterations
    :param current_time: last stored time
    :param i: current iteration number
    :return: the time when the last iteration was printed
    """
    if i % 5 == 0:
        new_time = time.time()
        elapsed_time = new_time - current_time
        current_time = new_time
        print(f'Iteration {i}. Elapsed time: {elapsed_time}')

    if i % 10 == 0:
        elapsed_time = time.time() - start_time
        print(f'Elapsed time since the beginning of the identification: {elapsed_time}')

    return current_time


# Helper functions related to the model identification
# %%
def optimize(A, b, optimization_method, **kwargs):
    """
    Method to carry out the optimization given the A and b matrices
    :param A: the A matrix that contains the equation components as a function of the cl and cd coefficients
    :param b: the thrust and torque values from the Matlab model
    :param optimization_method: method used for the optimization
    :param kwargs: variable that encapsulates in dictionary format all the variables used in the different optimizations
    It contains the LS_method, W_matrix, min_method, switch_constraints, degree_cla, degree_cda, min_angle, max_angle.
    :return: the cl and cd coefficients
    """

    if optimization_method == "LS":
        x = compute_LS(kwargs["LS_method"], kwargs["W_matrix"], A, b)
    elif optimization_method == "min":
        # The number of parameters that need to be identified
        n_params = A.shape[1]

        # The initial value of the cl and cd coefficients
        if kwargs["warm_starts"] is None:
            x0 = np.ones(n_params) * 0.1
            # x0 = np.array([0.44, 2.53, -9.19, 0.97, 0.1, 0.1]).T
        else:
            x0 = kwargs["warm_starts"]

        # Method used for the optimization
        min_method = kwargs["min_method"]

        # Creation of the constraints and optimization of the least squares equation
        switch_constraints = kwargs["switch_constraints"]
        if switch_constraints:
            arguments_constraint = (
                kwargs["degree_cla"], kwargs["degree_cda"], kwargs["min_angle"], kwargs["max_angle"])
            # noinspection SpellCheckingInspection
            constraints = ({"type": "ineq", "fun": nonlinear_constraint_drag_minimum, "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_maximum, "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_decaying, "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_positive_slope,
                            "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_alpha0, "args": arguments_constraint})
            x = minimize(minimize_func, x0, args=(A, b), method=min_method, constraints=constraints,
                         options={"disp": True, "maxiter": 2000, "ftol": 1e-08}).x
        else:
            x = minimize(minimize_func, x0, args=(A, b), method=min_method).x
    else:
        raise Exception("The proposed optimization method can not be executed.")
    return np.reshape(x, [-1, 1])


def minimize_func(x, A, b):
    """
    Function to minimize after having obtained the A and b matrices
    :param x: matrix to be found through the optimization
    :param A: matrix obtained that contains the values of the thrust and the torque from BEM as a function of the Cl and
    Cd coefficients
    :param b: matrix with the values of torque and thrust obtained from the Matlab
    :return: RMSE error that is used as optimization parameter
    """
    error = b - np.reshape(np.matmul(A, np.reshape(x, [-1, 1])), [-1, 1])
    error_thrust = error[::2]
    error_torque = error[1::2]
    RMSE_thrust = np.sqrt(np.mean(np.power(error_thrust, 2))) / np.std(b[::2])
    RMSE_torque = np.sqrt(np.mean(np.power(error_torque, 2))) / np.std(b[1::2])

    # Not chosen alternatives
    # RMSE_thrust = np.sqrt(np.mean(np.power(error_thrust, 2))) / (np.max(b[::2]) - np.min(b[::2]))
    # RMSE_torque = np.sqrt(np.mean(np.power(error_torque, 2))) / (np.max(b[1::2]) - np.min(b[1::2]))

    RMSE = (RMSE_thrust + RMSE_torque) / 2.
    # RMSE = np.sqrt(np.mean(np.power(error, 2))) / np.std(error)
    return RMSE


def nonlinear_constraint_drag_minimum(local_x, degree_cla, degree_cda, min_angle, max_angle):
    """
    Constraint to avoid that the drag value goes negative range
    :param local_x: value of the cl and cd coefficients
    :param degree_cla: degree of the lift coefficient equation
    :param degree_cda: degree of the drag coefficient equation
    :param min_angle: min angle of attack to consider for the constraint
    :param max_angle: max angle of attack to consider for the constraint
    :return: the minimum value in the cd curve
    """
    output = constraint_computation(-max_angle, max_angle, degree_cla, degree_cda, "cd", local_x, "min")
    return output


def nonlinear_constraint_lift_maximum(local_x, degree_cla, degree_cda, min_angle, max_angle):
    """
    Constraint to avoid that the lift value to obtain a very large value, namely higher than 5
    :param local_x: value of the cl and cd coefficients
    :param degree_cla: degree of the lift coefficient equation
    :param degree_cda: degree of the drag coefficient equation
    :param min_angle: min angle of attack to consider for the constraint
    :param max_angle: max angle of attack to consider for the constraint
    :return: the difference between the intended maximum of 3 and the actual cl curve
    """
    output = constraint_computation(-max_angle, max_angle, degree_cla, degree_cda, "cl", local_x, "max")
    return 5 - output


def nonlinear_constraint_lift_decaying(local_x, degree_cla, degree_cda, min_angle, max_angle):
    """
    Constraint to force the lift to decay with high angles of attack. dCl/da = x1 + 2*a*x2
    :param local_x: value of the cl and cd coefficients
    :param degree_cla: degree of the lift coefficient equation
    :param degree_cda: degree of the drag coefficient equation
    :param min_angle: min angle of attack to consider for the constraint
    :param max_angle: max angle of attack to consider for the constraint
    :return: the difference between the intended maximum of 3 and the actual cl curve
    """
    output = -constraint_computation(25, max_angle, degree_cla, degree_cda, "cl", local_x, "max", True)
    return output


def nonlinear_constraint_lift_positive_slope(local_x, degree_cla, degree_cda, min_angle, max_angle):
    """
    Constraint to force the lift to increase with the angle of attack for the range of 0-7 degrees
    :param local_x: value of the cl and cd coefficients
    :param degree_cla: degree of the lift coefficient equation
    :param degree_cda: degree of the drag coefficient equation
    :param min_angle: min angle of attack to consider for the constraint
    :param max_angle: max angle of attack to consider for the constraint
    :return: the difference between the intended maximum of 3 and the actual cl curve
    """
    output = constraint_computation(0, 7, degree_cla, degree_cda, "cl", local_x, "min", True)
    return output


def nonlinear_constraint_lift_alpha0(local_x, degree_cla, degree_cda, min_angle, max_angle):
    """
    Constraint to force the lift curve to cross the x-axis within the range [-10,10]. Since the airfoil is not
    excessively cambered, the angle of attack of zero lift would most likely be within this range. Usually it is between
    -3.5 and -1.5 when it is not symmetrical
    (http://www.aerodynamics4students.com/aircraft-performance/lift-and-lift-coefficient.php)
    :param local_x: value of the cl and cd coefficients
    :param degree_cla: degree of the lift coefficient equation
    :param degree_cda: degree of the drag coefficient equation
    :param min_angle: min angle of attack to consider for the constraint
    :param max_angle: max angle of attack to consider for the constraint
    :return: the difference between the intended maximum of 3 and the actual cl curve
    """
    output = -constraint_computation(-10, 10, degree_cla, degree_cda, "cl", local_x, "min")
    return output


def constraint_computation(min_angle, max_angle, degree_cla, degree_cda, cl_or_cd, local_x, min_or_max,
                           switch_slope=False):
    """
    Function to create the A matrix that will be used to defined the constraints by applying our knowledge about the
    cl and cd curves with respect to the angle of attack.
    :param min_angle: min angle of attack to consider for the constraint
    :param max_angle: max angle of attack to consider for the constraint
    :param degree_cla: degree of the lift coefficient equation
    :param degree_cda: degree of the drag coefficient equation
    :param cl_or_cd: whether we are talking about the cl or the cd alpha curves
    :param local_x: the local value of the cl and cd coefficients
    :param min_or_max: whether the minimum or maximum values should be considered
    :param switch_slope: whether the constraint uses the curve slope
    :return: the parameter that defines the constraint
    """

    angles = np.radians(np.arange(min_angle, max_angle + 1))
    local_A = np.zeros((angles.shape[0], degree_cla + degree_cda + 2))

    # Decide whether we are dealing with the lift or drag coefficient
    if cl_or_cd == "cl":
        if switch_slope:
            for i in range(1, degree_cla + 1):
                local_A[:, i] = i * np.power(angles, i - 1)
        else:
            for i in range(degree_cla + 1):
                local_A[:, i] = np.power(angles, i)
    elif cl_or_cd == "cd":
        for i in range(degree_cda + 1):
            local_A[:, 1 + degree_cla + i] = np.power(angles, i)
    else:
        raise Exception("This type of value range for accessing the A matrix of the constraints is not considered.")

    # Computation of the curve values
    local_b = np.matmul(local_A, np.reshape(local_x, [-1, 1]))

    # Choosing for the maximum or minimum value
    if min_or_max == "min":
        output = np.min(local_b)
    elif min_or_max == "max":
        output = np.max(local_b)
    else:
        raise Exception("This type of value range for accessing the A matrix of the constraints is not considered.")

    return output


def compute_LS(LS_method, W_matrix, A, b):
    """
    Compute the Least Squares method according to the method proposed in LS_method
    :param LS_method: method used for the computation of Least Squares, it can be Weighted Least Squares (WLS),
    Generalized Least Squares (GLS) or Ordinary Least Squares (OLS)
    :param W_matrix: the weight matrix for WLS
    :param A: the A matrix
    :param b: the b matrix
    :return: the parameter estimate vector
    """
    # Check what Least Squares method is used and apply the computation of the unknowns
    if LS_method == "WLS":  # Weighted Least Squares
        if W_matrix is None:
            W_straight = np.zeros(A.shape[0])
            W_straight[1::2] = 3600
            W_straight[0::2] = 1
            W_matrix = np.diag(W_straight)
        ATW = np.matmul(A.T, W_matrix)
        x = np.matmul(np.matmul(np.linalg.inv(np.matmul(ATW, A)), ATW), b)
    elif LS_method == "GLS":  # Generalized Least Squares
        x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
        error = b - np.matmul(A, x)
        sigma = np.matmul(error, error.T)
        ATS = np.matmul(A.T, np.linalg.inv(sigma))
        x = np.matmul(np.matmul(np.linalg.inv(np.matmul(ATS, A)), ATS), b)
    elif LS_method == "OLS":  # Ordinary Least Squares
        x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
    else:
        raise Exception("The provided Least Square method is not considered")

    return x


def compute_coeffs_grid_row(A, b, optimization_method, LS_method, W_matrix, degree_cla, degree_cda, min_angle,
                            max_angle, min_method, switch_constraints, number_samples_lst, filename_func,
                            activate_plotting=False, input_storage=None, warm_starts=None, current_coeffs_grid=None,
                            warm_start_row_index=0):
    """
    Function to compute the cl and cd coefficients for different number of data points but a constant number of blade
    sections
    :param A: the A matrix that contains the equation components as a function of the cl and cd coefficients
    :param b: the thrust and torque values from the Matlab model
    :param optimization_method: method used for the optimization
    :param LS_method: Least Squares method used: OLS, WLS, GLS
    :param W_matrix: the matrix used for WLS
    :param degree_cla: the polynomial degree of the Cl-alpha curve
    :param degree_cda: the degree of the polynomial that we want to use to approximate the Cd-a curve
    :param min_angle: the angle at which plotting starts
    :param max_angle: the angle at which plotting ends
    :param min_method: the method used for the constrained minimization problem
    :param switch_constraints: whether constraints should be used in the optimization
    :param number_samples_lst: the list with the number of samples
    :param filename_func: function for the creation of the filename name. A function is required since the number of
    blade sections is not provided as input
    :param activate_plotting: whether the plotting of the cla and inputs plots should be done
    :param input_storage: the dictionary with all the input information for each data point
    :param warm_starts: use already computed coefficients as warm start in the optimization
    :param current_coeffs_grid: current state of the coefficient grid
    :param warm_start_row_index: row index within the coefficient grid of the current row
    :return: array with the lift and drag coefficients identified with different numbers of samples
    """
    global figure_number

    # Creating the row for the coefficients' grid
    coeffs_grid_row = np.zeros((1, len(number_samples_lst), degree_cla+degree_cda+2))
    input_storage_local = {'ylabel': input_storage['ylabel'], 'title': input_storage['title']}
    for i, number_samples in enumerate(number_samples_lst):
        # Retrieving the information concerning the first "number_samples" data points
        A_local = A[:number_samples*2, :]
        b_local = b[:number_samples*2, :]

        # Compute the average warm start
        if warm_start_row_index != 0:
            warm_starts = (warm_starts + current_coeffs_grid[warm_start_row_index-1, i, :])/2

        # Carry out the optimization
        x = optimize(A_local, b_local, optimization_method, LS_method=LS_method, W_matrix=W_matrix, degree_cla=degree_cla,
                     degree_cda=degree_cda, min_angle=0, max_angle=max_angle,
                     min_method=min_method, switch_constraints=switch_constraints, warm_starts=warm_starts)
        warm_starts = x.flatten()

        # Add the optimised coefficients to the grid
        coeffs_grid_row[0, i, :] = np.reshape(x, [-1, ])
        if activate_plotting:
            # Restart the numbering such that not an infinite number of plots are created causing failure messages such
            # as "Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize"
            if figure_number > 50:
                figure_number = 1

            # Slicing the input info with the right amount of samples
            input_storage_local['body_velocity'] = input_storage['body_velocity'][:number_samples]
            input_storage_local['pqr'] = input_storage['pqr'][:number_samples]
            input_storage_local['omega'] = input_storage['omega'][:number_samples]

            # Do the plotting
            plot_cla(x, A_local, b_local, None, min_angle, max_angle, degree_cla, degree_cda)
            plot_inputs(input_storage_local)

            # Store the plots
            filename = filename_func(number_samples)
            multi_figure_storage(filename, figs=None, dpi=200)

            # Clean the figures. plt.close("all") is not enough because the information of the axis is not deleted and
            # remains in memory, so it has to be explicitly deleted
            for fig in plt.get_fignums():
                plt.figure(fig)
                plt.clf()
                plt.cla()
            plt.close('all')
    return coeffs_grid_row


def store_Abu_data(A, b, u, n_blade_segment, number_samples, va, min_method, name_suffix):
    """
    Function that stores the computed data points. NOT TESTED
    :param A: the A matrix of the LS
    :param b: the b matrix of the LS
    :param u: the input matrix of the LS
    :param n_blade_segment: number of blade segments
    :param number_samples: number of samples
    :param va: airspeed
    :param min_method: method used for the optimization
    :param name_suffix: suffix of the file name
    :return: None
    """
    name_lst = ["A", "b", "u"]
    extension = [".npy", ".npy", ""]
    variables = [A, b, u]
    new_samples = b.shape[0]/2
    for i, name in enumerate(name_lst):
        folder = 'Saved_data_points/' + name

        # Obtain the name of the file where the info will be saved
        desired_filename = f"{n_blade_segment}_bs_{number_samples}_dp_{va}_va_{min_method}_{name_suffix}__" + \
                           name + extension[i]
        desired_filename_lst = desired_filename.split("_")
        desired_filename_lst = desired_filename_lst[:2]+desired_filename_lst[3:-2]

        # Check if the file exists
        filenames = os.listdir(folder)
        filenames_lst = [filename.split("_") for filename in filenames]
        filenames_lst_lst = [filename_lst[:2] + filename_lst[3:-2] for filename_lst in filenames_lst]
        check_filename_existence = desired_filename_lst in filenames_lst_lst

        # If file exists, add already existing data points
        if check_filename_existence:
            index = filenames_lst_lst.index(desired_filename_lst)
            available_samples = int(filenames_lst[index][2])
            total_samples = int(available_samples + new_samples)
            old_filename = os.path.join(folder, filenames[index])
            desired_filename = f"{n_blade_segment}_bs_{total_samples}_dp_{va}_va_{min_method}_{name_suffix}__" + \
                               name + extension[i]
            if extension[i] == ".npy":
                stored_variable = np.load(old_filename)
                variable = np.vstack((stored_variable, variables[i]))
            else:
                dbfile = open(old_filename, 'rb')
                stored_variable = pickle.load(dbfile)
                variable = stored_variable.copy()
                dbfile.close()
                variable['body_velocity'] += variables[i]['body_velocity']
                variable['pqr'] += variables[i]['pqr']
                variable['omega'] += variables[i]['omega']
        else:
            variable = variables[i]

        with open(folder + "/" + desired_filename, 'wb') as f:
            if extension[i] == ".npy":
                np.save(f, variable)
            else:
                pickle.dump(variable, f)
        if check_filename_existence:
            os.remove(old_filename)


def check_Abu_data(n_blade_segment, number_samples, va, min_method, name_suffix):
    """
    Function to check the number of samples available and the number of samples to still be computed.
    :param n_blade_segment: number of blade sections
    :param number_samples: the total number of samples desired
    :param va: the airspeed
    :param min_method: the method used for optimization
    :param name_suffix: component added at the end of the filename
    :return: the number of available files within the desired file and the number of samples that still need to be
    computed
    """
    folder = 'Saved_data_points/b'

    # Obtain the name of the file where the info will be saved
    desired_filename = f"{n_blade_segment}_bs_{number_samples}_dp_{va}_va_{min_method}_{name_suffix}__b.npy"

    desired_filename_lst = desired_filename.split("_")
    desired_filename_lst = desired_filename_lst[:2] + desired_filename_lst[3:-2]

    # Check if the file exists
    filenames_lst = [filename.split("_") for filename in os.listdir(folder)]
    filenames_lst_lst = [filename_lst[:2] + filename_lst[3:-2] for filename_lst in filenames_lst]
    check_filename_existence = desired_filename_lst in filenames_lst_lst

    # If file exists, retrieve the number of data points already computed
    if check_filename_existence:
        index = filenames_lst_lst.index(desired_filename_lst)
        available_samples = int(filenames_lst[index][2])
        return available_samples, max(number_samples-available_samples, 0)
    return 0, number_samples


def retrieve_Abu_data(n_blade_segment, number_samples, va, min_method, name_suffix):
    """
    Function to retrieve samples from the saved files.
    :param n_blade_segment: number of blade sections
    :param number_samples: the total number of samples desired
    :param va: the airspeed
    :param min_method: the method used for optimization
    :param name_suffix: component added at the end of the filename
    :return: the A, b and u arrays of the chosen file
    """
    name_lst = ["A", "b", "u"]
    extension = [".npy", ".npy", ""]
    variables = []
    for i, name in enumerate(name_lst):
        folder = 'Saved_data_points/' + name

        # Obtain the name of the file where the info will be saved
        desired_filename = f"{n_blade_segment}_bs_{number_samples}_dp_{va}_va_{min_method}_{name_suffix}__" + \
                           name + extension[i]
        desired_filename_lst = desired_filename.split("_")
        desired_filename_lst = desired_filename_lst[:2]+desired_filename_lst[3:-2]

        # Check if the file exists
        filenames = os.listdir(folder)
        filenames_lst = [filename.split("_") for filename in filenames]
        filenames_lst_lst = [filename_lst[:2] + filename_lst[3:-2] for filename_lst in filenames_lst]
        check_filename_existence = desired_filename_lst in filenames_lst_lst

        # If the file does not exist, return error
        if not check_filename_existence:
            raise Exception(f"The searched file ({desired_filename_lst} does not exist.")

        # If file exists, return already existing data points
        index = filenames_lst_lst.index(desired_filename_lst)
        old_filename = os.path.join(folder, filenames[index])
        if extension[i] == ".npy":
            stored_variable = np.load(old_filename)
            variable = stored_variable[:2*number_samples, :]
        else:
            dbfile = open(old_filename, 'rb')
            stored_variable = pickle.load(dbfile)
            variable = stored_variable
            dbfile.close()
            variable['body_velocity'] = variable['body_velocity'][:number_samples]
            variable['pqr'] = variable['pqr'][:number_samples]
            variable['omega'] = variable['omega'][:number_samples]
        variables.append(variable)

    return variables[0], variables[1], variables[2]


def personal_opt(func, x0, den):
    """
    Personal optimization (gradient descend) function to substitute the scipy.optimize.minimize Nelder_Mead function
    used for the computation of the induced velocity. The function stops when the denominator becomes zero, when the
    maximum number of iterations has been reached or when the optimization variable has barely changed in 20 iterations.
    This function takes less time than the scipy counterpart and it can be implemented in C++.
    :param func: function to minimize
    :param x0: initial condition
    :param den: the denominator of the derivative of the func function
    :return: the solution of the function func
    """
    x = x0
    alpha = 0.5  # the learning rate
    th = 0.01  # threshold to stop the iterative process
    counter = 0  # current counter number
    previous_der = func(x)  # value of the function at the initial condition
    for i in range(10000):
        # If the denominator of the function is zero, then we have reached the desired point
        if den([x]) < 1e-10:
            return x

        # Compute and apply the gradient
        der = func(x)
        x_new = x - alpha*der

        # If the derivative changes sign, it means that we have passed through the minimum. Reduce alpha
        if der != 0:
            if previous_der/der < 0:
                alpha = alpha/2.

        # If there has not been a change in x, return function
        step = x_new-x
        x = x_new
        previous_der = der
        if abs(step) < th:
            counter += 1
            if counter > 20:
                return x
        else:
            counter = 0
    return x


# Forces and moments computation
#%
def FM_time_simulation(propeller, propeller_func, propeller_func_input, mass_aero="a", more_print="", switch_plot=True):
    """
    Function to run a time-based simulation of some type of forces and moments. The type is determined by the
    propeller_func provided as input
    :param propeller: propeller object
    :param propeller_func: propeller function used for the generation of forces and moments
    :param propeller_func_input: dictionary containing all the potential inputs to the propeller_func
    :param mass_aero: the type of plot labels
    :param more_print: whether some additional text should be included in the progress message
    :param switch_plot: whether plots should be generated
    :return: arrays with the forces and moments experienced at every time step
    """
    def get_input_names(func):
        """
        Get arguments names from function
        :param func: function from which the arguments shall be retrieved
        :return: the function arguments
        """
        return inspect.getfullargspec(func)[0]

    def filter_dict(dict_, keys):
        """
        Given a dictionary, filter key-value combinations given some keys
        :param dict_: dictionary that should be filtered
        :param keys: keys that should be used for filtering
        :return: the new filtered dictionary
        """
        return {k: dict_[k] for k in keys if k != "self"}

    # Required parameters are extracted from the input dictionary
    n_points = propeller_func_input["n_points"]
    rotation_angle = propeller_func_input["rotation_angle"]
    omega = propeller_func_input["omega"]
    total_time = propeller_func_input["total_time"]
    dt = propeller_func_input["dt"]

    # Time-based simulation
    F_lst = np.zeros((3, n_points))
    M_lst = np.zeros((3, n_points))
    rotation_angle_lst = np.zeros(n_points)  # the rotation angle of the propeller is started at 0
    propeller.set_rotation_angle(rotation_angle)

    # Extract the parameters of the dict that are useful for the chosen propeller function
    filtered_propeller_func_input = filter_dict(propeller_func_input, get_input_names(propeller_func))
    for i in range(n_points):
        if not i % 10:
            print(f'{more_print}Iteration {i} out of {n_points - 1}')
        F, M = propeller_func(**filtered_propeller_func_input)
        F_lst[:, i] = F.flatten()
        M_lst[:, i] = M.flatten()
        rotation_angle_lst[i] = rotation_angle
        rotation_angle = propeller.update_rotation_angle(omega, dt)  # with every time step, update the prop rotation

    # Plotting the forces and moments
    if switch_plot:
        plot_FM(np.arange(0, total_time + dt, dt), rotation_angle_lst, F_lst, M_lst, mass_aero=mass_aero)
    return F_lst, M_lst


# Plotters
# %%
def plot_chord_twist(chord, twist):
    """
    Two plots in a single figure:
        - The first plot shows the chord of the blade
        - The second plot shows the twist of the blade along its span

    This figure is Figure 9.11 in the thesis.
    :param chord: list of chord values
    :param twist: list of twist values
    :return: None
    """
    global figure_number

    x_chord = range(len(chord))
    y_chord_1 = [i / 2 for i in chord]
    y_chord_2 = [-i for i in y_chord_1]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(x_chord, y_chord_1, 'r-', linewidth=4)
    plt.plot(x_chord, y_chord_2, 'r-', linewidth=4)
    plt.grid(True)

    y_chord_3 = [i*1000 for i in chord]  # transform values from meters to mm
    x_twist = range(len(twist))  # compute twist values
    y_twist = [degrees(i) for i in twist]  # transform twist values to degrees

    # Plot the chord values along the blade span. The commented out l3 is a marker at the location of the maximum chord
    plt.figure(figure_number)
    figure_number += 1
    ax1 = plt.gca()
    l1 = plt.plot(x_chord, y_chord_3, color='#1f77b4', linestyle='-', linewidth=4)
    l2 = plt.plot(x_chord[0], 13, color='#1f77b4', marker='o', markersize=20, linewidth=0)
    # l3 = plt.plot(x_chord[y_chord_3.index(np.max(y_chord_3))], 20, color='#1f77b4', marker='X', markersize=20, linewidth=0)
    l4 = plt.plot(x_chord[-1], 8, color='#1f77b4', marker='v', markersize=20, linewidth=0)
    plt.ylabel("$c$ [mm]")
    plt.ylim([0, 24])
    plt.yticks(np.arange(0, 28, step=4))
    plt.xlabel("Blade section number [-]")
    ax1.tick_params('y', colors='#1f77b4')
    ax1.yaxis.label.set_color('#1f77b4')
    plt.grid(True)

    # Plot the twist values along the blade span
    ax2 = ax1.twinx()
    l5 = ax2.plot(x_twist, y_twist, color='#ff7f0e', linestyle='--', linewidth=4)
    ax2.set_ylabel("$\\theta$ [deg]")
    ax2.set_ylim([0,30])
    ax2.tick_params('y', colors='#ff7f0e')
    ax2.yaxis.label.set_color('#ff7f0e')
    plt.grid(True)
    # plt.legend([l1[0],l5[0], l2[0], l3[0], l4[0]], ["Chord", "Twist", '$c_r$=13 mm', '$c_c$=20 mm', '$c_t$=8 mm'], loc=8)
    plt.legend([l1[0], l5[0], l2[0], l4[0]], ["Chord", "Twist", '$c_r$',  '$c_t$'], loc=3)


def plot_cla(x, A, b, aoa_storage, start_alpha, finish_alpha, degree_cla, degree_cda):
    """
    Function that plots the approximated cl-alpha and cd-alpha curves, and the verification of those approximations. In
    that regard, it plots the thrust and torque absolute errors and their autocorrelations. It also plots the average
    angle of attack seen by each blade element in the form of a box plot
    :param x: vector of unknown states
    :param A: regression matrix
    :param b: observation vector
    :param aoa_storage:
    :param start_alpha: first angle of attack to plot
    :param finish_alpha: last angle of attack to plot
    :param degree_cla: degree of the cl-alpha polynomial
    :param degree_cda: degree of the cd-alpha polynomial
    :return: None
    """
    global figure_number

    def cla_equation(alpha):
        """
        Creates the data points for the cl-alpha curve given an angle of attach and the cl-alpha coefficients
        :param alpha: angle of attack
        :return: returns the value of cl given an angle of attack
        """
        cl = 0
        for i in range(degree_cla + 1):
            cl += alpha ** i * x[i]
        return cl

    def cda_equation(alpha):
        """
        Creates the data points for the cd-alpha curve given an angle of attach and the cd-alpha coefficients
        :param alpha: angle of attack
        :return: returns the value of cd given an angle of attack
        """
        cd = 0
        for i in range(degree_cla + 1, degree_cla + degree_cda + 2):
            exponent = i - (degree_cla + 1)
            cd += alpha ** exponent * x[i]
        return cd

    # Creation of the range of alphas to plot and the cl-cd coefficients
    alphas = np.arange(start_alpha, finish_alpha, 0.1)
    cls = [cla_equation(radians(aoa)) for aoa in alphas]
    cds = [cda_equation(radians(aoa)) for aoa in alphas]

    # Obtain the title for the cl-alpha plot
    title = "Cl-alpha curve: Cl = "
    for i in range(degree_cla + 1):
        title += f'{np.round(x[i].item(),2)} $\\alpha^{i}$'
        if i != degree_cla:
            if x[i+1].item() > 0:
                title += '+'

    # Plot the cl-alpha curve. For the final formatted cl-alpha plot found in the thesis and paper,
    # go to ClCd_plotter.py
    fig = plt.figure(figure_number)
    figure_number += 1
    plt.plot(alphas, cls, 'r-', linewidth=4)
    plt.xlabel("$\\alpha$ [deg]")
    plt.ylabel("$C_l$ [-]")
    # plt.title(title)  # commented out for the paper plots
    ax = plt.gca()
    if (max(cls) - min(cls)) / 0.1 > 20:  # different discretizations of the y-axis depending on y values encountered
        y_discretisation = 0.5
    elif (max(cls) - min(cls)) / 0.1 > 2:
        y_discretisation = 0.2
    else:
        y_discretisation = 0.01
    ax.yaxis.set_major_locator(MultipleLocator(y_discretisation))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.grid(True)
    fig.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)

    # Plot the cd-alpha curve. For the final formatted cd-alpha plot found in the thesis and paper,
    # go to ClCd_plotter.py
    title = "Cd-alpha curve: Cd = "
    for i in range(degree_cla + 1, degree_cla + degree_cda + 2):
        title += f'{np.round(x[i].item(),2)} $\\alpha^{i-(degree_cla + 1)}$'
        if i != degree_cla + degree_cda + 1:
            if x[i+1].item() > 0:
                title += '+'
    fig = plt.figure(figure_number)
    figure_number += 1
    plt.plot(alphas, cds, 'r-', linewidth=4)
    plt.xlabel("$\\alpha$ [deg]")
    plt.ylabel("$C_d$ [-]")
    # plt.title(title)
    ax = plt.gca()
    if (max(cds) - min(cds)) / 0.1 > 20:
        y_discretisation = 0.5
    elif (max(cds) - min(cds)) / 0.1 > 2:
        y_discretisation = 0.1
    else:
        y_discretisation = 0.01
    ax.yaxis.set_major_locator(MultipleLocator(y_discretisation))  # y-axis discretisation
    ax.xaxis.set_major_locator(MultipleLocator(5))  # x-axis discretization
    plt.grid(True)
    fig.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)

    # Compare the results predicted by multiplying A and x, with respect to the observations in b for the thrust
    b_approx = np.reshape(np.matmul(A, x), [-1, 1])  # predicted observed data resulting from identified x vector
    number_p = A.shape[0]
    data_points = range(int(number_p / 2))
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, b[::2], 'ro')  # color measured values with red
    plt.plot(data_points, b_approx[::2], 'bo')  # color approximate values with blue
    plt.xlabel("Data point [-]")
    plt.ylabel("Thrust value")
    plt.title("Thrust: Ax vs b")
    plt.grid(True)

    # Compare the results predicted by multiplying A and x, with respect to the observations in b for the torque
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, b[1::2], 'ro')  # color measured values with red
    plt.plot(data_points, b_approx[1::2], 'bo')  # color approximate values with blue
    plt.xlabel("Data point [-]")
    plt.ylabel("Torque value")
    plt.title("Torque: Ax vs b")
    plt.grid(True)

    # Computation of error
    error = b - b_approx
    error_T = error[::2]  # absolute error in thrust
    error_Q = error[1::2]  # absolute error in torque
    average_error_T = np.mean(error_T)  # mean thrust error
    average_error_Q = np.mean(error_Q)  # mean torque error
    percentage_error = [error[i, 0] / b[i, 0] * 100 for i in range(error.shape[0])]  # relative error
    print(f"Mean error thrust: {average_error_T} [N].")
    print(f"Mean error torque: {average_error_Q} [Nm].")
    print(f"Maximum error percentage: {max(percentage_error)}%")

    # Durbin Watson autocorrelation statistical test
    DW_T = durbin_watson(error_T)  # for thrust
    DW_Q = durbin_watson(error_Q)  # for torque
    print(f"Durbin Watson test for thrust: {np.round(DW_T[0],3)}")
    print(f"Durbin Watson test for torque: {np.round(DW_Q[0],3)}")

    # Computation of relative or normalized RMSE
    # Divide RMSE by difference between max and min of observed values
    RMSE_T1 = np.sqrt(np.mean(np.power(error_T, 2))) / (np.max(b[::2]) - np.min(b[::2]))
    RMSE_Q1 = np.sqrt(np.mean(np.power(error_Q, 2))) / (np.max(b[1::2]) - np.min(b[1::2]))

    # Divide RMSE by standard deviation of observed values
    RMSE_T2 = np.sqrt(np.mean(np.power(error_T, 2))) / np.std(b[::2])
    RMSE_Q2 = np.sqrt(np.mean(np.power(error_Q, 2))) / np.std(b[1::2])

    # THRUST
    # Print RMSE error for thrust
    text_T_RMSE = f'Diff RMSE = {np.round(RMSE_T1, 2)} and NRMSE$_\\tau$ = {np.round(RMSE_T2, 2)}'
    print(text_T_RMSE)

    # Obtain legend for thrust error
    average_error_T_scientific = np.format_float_scientific(average_error_T,2)  # average T error in scientific form
    index_e = average_error_T_scientific.index("e")
    T_mean_label_number = average_error_T_scientific[:index_e] + \
                          f"$\\cdot$10$^{{{int(average_error_T_scientific[index_e + 1:])}}}$"  # label in 10^x notation
    T_mean_text = f'Mean = {T_mean_label_number} [N]'

    # Plot thrust absolute error and mean. Figures 9.14 and 9.15
    fig = plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, error[::2], 'g-', alpha=0.5)
    plt.plot(data_points, np.repeat(average_error_T, len(data_points)), 'k--', label=T_mean_text, linewidth=3)
    plt.xlabel("Data point [-]")
    plt.ylabel("$\\epsilon_\\tau$ [N]")
    # plt.title("Approximation error thrust")  # comment out title for paper plots
    plt.legend(loc=3)
    plt.grid(True)
    fig.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)

    # Validation plot to observe the whiteness of the residual. Figures 9.18, 9.19 in the thesis and Figs. 60 and 61 of
    # paper: "Blade Element Theory Model for UAV Blade Damage Simulation"
    conf = 1.96 / np.sqrt(error_T.shape[0])  # 95% confidence interval of normally distributed data
    fig = plt.figure(figure_number)
    figure_number += 1
    ax = plt.gca()
    ax.acorr(np.reshape(error_T, [-1, ]), maxlags=None, usevlines=False, normed=True, linestyle="-", marker='',
             color="b", linewidth=3)  # Compute the correlations
    plt.axhline(y=conf, xmin=-error_T.shape[0], xmax=error_T.shape[0], color="red", linestyle="--", linewidth=3,
                label="95% confidence bounds")  # compute the 95% confidence bounds
    plt.axhline(y=-conf, xmin=-error_T.shape[0], xmax=error_T.shape[0], color="red", linestyle="--", linewidth=3)
    plt.xlabel("Number of lags [-]")
    plt.ylabel("Normalised $\\epsilon_\\tau$ autocorrelation [-]")
    # plt.title("Autocorrelation of model residual: Thrust")  # comment out title for paper plots
    plt.grid(True)
    plt.legend(loc='upper right')
    fig.subplots_adjust(left=0.12, top=0.95, right=0.99, bottom=0.13)

    # TORQUE
    # Print RMSE error for torque
    text_Q_RMSE = f'Diff RMSE = {np.round(RMSE_Q1, 2)} and NRMSE$_Q$ = {np.round(RMSE_Q2, 2)}'
    print(text_Q_RMSE)

    # Obtain legend for torque error
    average_error_Q_scientific = np.format_float_scientific(average_error_Q,2)  # average Q error in scientific form
    index_e = average_error_Q_scientific.index("e")
    Q_mean_label_number = average_error_Q_scientific[:index_e] + \
                          f"$\\cdot$10$^{{{int(average_error_Q_scientific[index_e + 1:])}}}$"  # label in 10^x notation
    Q_mean_text = f'Mean = {Q_mean_label_number} [Nm]'

    # Plot of error for torque. Figures 9.16, 9.17
    fig = plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, error[1::2], 'g-', alpha=0.5)
    plt.plot(data_points, np.repeat(average_error_Q, len(data_points)), 'k--', label=Q_mean_text, linewidth=3)
    plt.xlabel("Data point [-]")
    plt.ylabel("$\\epsilon_Q$ [Nm]")
    # plt.title("Approximation error torque")  # comment out title for paper plots
    plt.grid(True)
    plt.legend(loc=3)
    fig.subplots_adjust(left=0.14, top=0.95, right=0.99, bottom=0.13)

    # Validation plot to observe the whiteness of the residuals. Figures 9.20, 9.21 in the thesis and Figs. 62 and 63 of
    # paper: "Blade Element Theory Model for UAV Blade Damage Simulation"
    fig = plt.figure(figure_number)
    ax = plt.gca()
    figure_number += 1
    ax.acorr(np.reshape(error_Q, [-1, ]), maxlags=None, usevlines=False, normed=True, linestyle="-", marker='',
             color="blue", linewidth=3)
    plt.axhline(y=conf, xmin=-error_Q.shape[0], xmax=error_Q.shape[0], color="red", linestyle="--", linewidth=3,
                label="95% confidence bounds")
    plt.axhline(y=-conf, xmin=-error_Q.shape[0], xmax=error_Q.shape[0], color="red", linestyle="--", linewidth=3)
    plt.xlabel("Number of lags [-]")
    plt.ylabel("Normalised $\\epsilon_Q$ autocorrelation [-]")
    # plt.title("Autocorrelation of model residual: Torque")
    plt.grid(True)
    plt.legend(loc='upper right')
    fig.subplots_adjust(left=0.12, top=0.95, right=0.99, bottom=0.13)

    # ANGLE OF ATTACK
    # Plot of the angles of attack seen by each of the selected blade sections. It is represented as a box plot such
    # that it can be seen the average angle of attack, as well as the range of alphas seen by its section. Figures 9.38,
    # 9.39 in the thesis or Fig. 31 and 32 in paper: "Blade Element Theory Model for UAV Blade Damage Simulation"
    if aoa_storage is not None:
        n_blade_sections = len(aoa_storage.keys())
        n_aoa = len(aoa_storage[0])
        flag = True
        while flag:  # keep plotting until user decides to stop plotting
            user_input = input(f'Out of {n_blade_sections} blade sections, which ones would you like to plot? '
                               f'Please give the start and end sections separated by a comma.')

            # If the user uses the letter s, it means that it wants to stop without plotting.
            if user_input == "s":
                flag = False
                continue

            # Check that feasible blade sections have been given
            user_input = list(map(int, user_input.split(',')))
            start_section, end_section = user_input
            flag = False
            if start_section < 0 or end_section >= n_blade_sections:
                print("Those blade section indices can not be applied")
                continue

            # Carry out the counting of the angle of attack seen by the blade sections
            n_sections = end_section - start_section + 1
            aoa_vectors = np.zeros((n_aoa, n_sections))
            for i in range(start_section, end_section + 1):
                counter = i - start_section
                aoa_vectors[:, counter] = np.reshape(np.array(aoa_storage[i]), [-1, ])

            # Box plot of the angle of attack as seen by each of the blade sections
            fig = plt.figure(figure_number)
            ax = plt.gca()
            figure_number += 1
            plt.boxplot(np.degrees(aoa_vectors), showfliers=False, patch_artist=True)
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            plt.ylabel(r"$\alpha$ [deg]")
            plt.xlabel("Blade section number [-]")
            plt.grid(True, alpha=0.5)
            fig.subplots_adjust(left=0.09, top=0.95, right=0.97, bottom=0.13)
            plt.show()


def plot_inputs(inputs_dict):
    """
    Function to plot the inputs used for the data point generation
    :param inputs_dict: dictionary with the value of the inputs used
    :return: None
    """
    global figure_number
    inputs = inputs_dict.keys()
    for key in inputs:
        if key != "ylabel" and key != "title":
            if type(inputs_dict[key][0]) != float:  # 3 subplots for variables with values along each axis (e.g. Vb)
                fig = plt.figure(figure_number)
                figure_number += 1
                axes = fig.subplots(3, 1, gridspec_kw={'wspace': 0.5,'hspace': 0.5})
                axis_names = ["x", "y", "z"]
                counter = 0
                for ax in axes:
                    ax.plot(np.reshape(np.array(inputs_dict[key])[:, counter, :], [-1,]), "bo")
                    ax.set_xlabel("Data point number [-]")
                    ylabel = f'{inputs_dict["ylabel"][0][key]} for {axis_names[counter]}-axis'
                    ax.set_ylabel(ylabel)
                    ax.grid(True)
                    counter += 1
                fig.suptitle(inputs_dict["title"][0][key])
            else:  # 1 plot for variables with a single vaue (e.g. \omega)
                plt.figure(figure_number)
                figure_number += 1
                plt.plot(inputs_dict[key], "bo")
                plt.xlabel("Data point number [-]")
                plt.ylabel(inputs_dict["ylabel"][0][key])
                plt.title(inputs_dict["title"][0][key])
                plt.grid(True)
                plt.show()


def plot_coeffs_map(coeffs_grid, degree_cla, degree_cda, x_coords, y_coords, switch_title=True):
    """
    Function to plot the coefficients for different blade discretisations and number of data points. The plots
    correspond to Figures D.1 and D.2 in the thesis
    :param coeffs_grid: identified cl and cd coefficients
    :param degree_cla: degree of cl polynomial
    :param degree_cda: degree of cd polynomial
    :param x_coords: the values of the x-axis
    :param y_coords: the values of the y-axis
    :param switch_title: whether the plots should have a title
    :return: None
    """
    global figure_number

    # Computation of the axis coords
    x_step = x_coords[1]-x_coords[0]
    y_step = y_coords[1]-y_coords[0]
    x_coords_edges = np.arange(x_coords[0]-x_step/2, x_coords[-1] + x_step, x_step)
    y_coords_edges = np.arange(y_coords[0] - y_step / 2, y_coords[-1] + y_step, y_step)
    x_coords_mesh = np.tile(np.reshape(x_coords_edges, [1, -1]), (len(y_coords) + 1, 1))
    y_coords_mesh = np.tile(np.reshape(y_coords_edges, [-1, 1]), (1, len(x_coords) + 1))

    # Plot the cl coefficients
    coeff_plotter(coeffs_grid[:, :, :degree_cla+1], degree_cla, "lift", x_coords_mesh, y_coords_mesh, gradient=False,
                  cbar_label_func=lambda c: f"$x_{c} \; [-]$", switch_title=switch_title)

    # Plot the cd coeffs
    coeff_plotter(coeffs_grid[:, :, degree_cla+1:], degree_cda, "drag", x_coords_mesh, y_coords_mesh, gradient=False,
                  cbar_label_func=lambda c: f"$y_{c} \; [-]$", switch_title=switch_title)


def plot_derivative_coeffs_map(coeffs_grid, degree_cla, degree_cda, x_coords, y_coords, switch_title=True):
    """
    Function to plot the percentage change of the coefficients with respect to changes in the number of samples and the
    blade discretisation. The plots correspond to Figures D.3 and D.4 in the thesis.
    :param coeffs_grid: identified cl and cd coefficients
    :param degree_cla: degree of cl polynomial
    :param degree_cda: degree of cd polynomial
    :param x_coords: the values of the x-axis
    :param y_coords: the values of the y-axis
    :param switch_title: whether the plots should have a title
    :return: None
    """
    # Computation of the axis coords
    x_step = x_coords[1]-x_coords[0]
    y_step = y_coords[1]-y_coords[0]

    x_coords_edges = np.arange(x_coords[0] - x_step / 2, x_coords[-1] + x_step, x_step)
    x_coords_edges_mod = np.arange(x_coords[1] - x_step / 2, x_coords[-1] + x_step, x_step)
    y_coords_edges = np.arange(y_coords[0] - y_step / 2, y_coords[-1] + y_step, y_step)
    y_coords_edges_mod = np.arange(y_coords[1] - y_step / 2, y_coords[-1] + y_step, y_step)

    x_coords_mesh_der_x = np.tile(np.reshape(x_coords_edges_mod, [1, -1]), (len(y_coords) + 1, 1))
    x_coords_mesh_der_y = np.tile(np.reshape(x_coords_edges, [1, -1]), (len(y_coords), 1))
    y_coords_mesh_der_x = np.tile(np.reshape(y_coords_edges, [-1, 1]), (1, len(x_coords)))
    y_coords_mesh_der_y = np.tile(np.reshape(y_coords_edges_mod, [-1, 1]), (1, len(x_coords) + 1))

    # Computation of the derivative coefficients
    # Along the x_axis
    coeffs_grid_der_x = (coeffs_grid[:, 1:, :]-coeffs_grid[:, :-1, :]) / \
                        np.tile(coeffs_grid[:, [-1], :], (1, coeffs_grid.shape[1]-1, 1)) * 100
    coeffs_grid_der_y = (coeffs_grid[1:, :, :] - coeffs_grid[:-1, :, :]) / \
                        np.tile(coeffs_grid[[-1], :, :], (coeffs_grid.shape[0] - 1, 1, 1)) * 100

    # Derivative along the number of data samples axis
    # Plot the cl coefficients
    coeff_plotter(coeffs_grid_der_x[:, :, :degree_cla+1], degree_cla, "percentage change of the lift",
                  x_coords_mesh_der_x, y_coords_mesh_der_x, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$D_q x_{{{c}}}\; [\\%]$", switch_title=switch_title)

    # Plot the cd coeffs
    coeff_plotter(coeffs_grid_der_x[:, :, degree_cla+1:], degree_cda, "percentage change of the drag",
                  x_coords_mesh_der_x, y_coords_mesh_der_x, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$D_q y_{{{c}}}\; [\\%]$", switch_title=switch_title)

    # Derivative along the number of blade sections axis
    # Plot the cl coefficients
    coeff_plotter(coeffs_grid_der_y[:, :, :degree_cla+1], degree_cla, "percentage change of the lift",
                  x_coords_mesh_der_y, y_coords_mesh_der_y, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$D_{{n_{{bs}}}} x_{{{c}}}\; [\\%]$", switch_title=switch_title)

    # Plot the cd coeffs
    coeff_plotter(coeffs_grid_der_y[:, :, degree_cla+1:], degree_cda, "percentage change of the drag",
                  x_coords_mesh_der_y, y_coords_mesh_der_y, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$D_{{n_{{bs}}}} y_{{{c}}}\; [\\%]$", switch_title=switch_title)

    # Using the maximum
    # Plot the maximum values from the coefficients axis
    coeffs_grid_der_x_max = np.reshape(np.amax(np.abs(coeffs_grid_der_x), axis=2),
                                       [coeffs_grid_der_x.shape[0], coeffs_grid_der_x.shape[1], 1])
    coeffs_grid_der_y_max = np.reshape(np.amax(np.abs(coeffs_grid_der_y), axis=2),
                                       [coeffs_grid_der_y.shape[0], coeffs_grid_der_y.shape[1], 1])

    coeff_plotter(coeffs_grid_der_x_max, 0, "percentage change of the maximum",
                  x_coords_mesh_der_x, y_coords_mesh_der_x, gradient=True, switch_title=switch_title)

    coeff_plotter(coeffs_grid_der_y_max, 0, "percentage change of the maximum",
                  x_coords_mesh_der_y, y_coords_mesh_der_y, gradient=True, switch_title=switch_title)

    # Final plot merging both
    coeffs_grid_der_max = np.maximum(coeffs_grid_der_x_max[1:, :, :], coeffs_grid_der_y_max[:, 1:, :])
    coeff_plotter(coeffs_grid_der_max, 0, "percentage change of the maximum",
                  x_coords_mesh_der_x[1:, :], y_coords_mesh_der_y[:, 1:], gradient=True, switch_title=switch_title)

    coeffs_grid_der_sum = coeffs_grid_der_x_max[1:, :, :] + coeffs_grid_der_y_max[:, 1:, :]
    coeff_plotter(coeffs_grid_der_sum, 0, "percentage change of the sum",
                  x_coords_mesh_der_x[1:, :], y_coords_mesh_der_y[:, 1:], gradient=True, switch_title=switch_title)


def plot_MA_derivative_coeffs_map(coeffs_grid, degree_cla, degree_cda, x_coords, y_coords, horizon_length, threshold,
                                  switch_title=True):
    """
    Function to plot the percentage change of the coefficients with respect to changes in the number of samples and the
    blade discretisation. This plots correspond to Figures D.6 and D.7 in the thesis.
    :param coeffs_grid: identified cl and cd coefficients
    :param degree_cla: degree of cl polynomial
    :param degree_cda: degree of cd polynomial
    :param x_coords: the values of the x-axis
    :param y_coords: the values of the y-axis
    :param horizon_length: length of the horizon used for the moving average
    :param threshold: the threshold used to determine the right number of data points and number of blades
    :param switch_title: whether the plots need to have title
    :return: None
    """
    # Computation of the axis coords
    x_step = x_coords[1]-x_coords[0]
    y_step = y_coords[1]-y_coords[0]

    x_coords_edges = np.arange(x_coords[0] - x_step / 2, x_coords[-1] + x_step, x_step)
    x_coords_edges_mod = np.arange(x_coords[1] - x_step / 2, x_coords[-1] + x_step, x_step)
    y_coords_edges = np.arange(y_coords[0] - y_step / 2, y_coords[-1] + y_step, y_step)
    y_coords_edges_mod = np.arange(y_coords[1] - y_step / 2, y_coords[-1] + y_step, y_step)

    x_coords_mesh_der_x = np.tile(np.reshape(x_coords_edges_mod, [1, -1]), (len(y_coords) + 1, 1))
    x_coords_mesh_der_y = np.tile(np.reshape(x_coords_edges, [1, -1]), (len(y_coords), 1))
    y_coords_mesh_der_x = np.tile(np.reshape(y_coords_edges, [-1, 1]), (1, len(x_coords)))
    y_coords_mesh_der_y = np.tile(np.reshape(y_coords_edges_mod, [-1, 1]), (1, len(x_coords) + 1))

    # Computation of the moving average of the derivative of the coefficients
    # Along the x_axis
    coeffs_grid_der_x = (coeffs_grid[:, 1:, :] - coeffs_grid[:, :-1, :]) / np.tile(coeffs_grid[:, [-1], :], (1, coeffs_grid.shape[1]-1, 1)) * 100
    MA_coeffs_grid_der_x = np.zeros(coeffs_grid_der_x.shape)
    for row in range(coeffs_grid_der_x.shape[0]):
        for column in range(coeffs_grid_der_x.shape[1]):
            for coeff in range(coeffs_grid_der_x.shape[2]):
                element = np.mean(coeffs_grid_der_x[row, max(0, column - horizon_length):column + 1, coeff])
                MA_coeffs_grid_der_x[row, column, coeff] = element

    # Along the y_axis
    coeffs_grid_der_y = (coeffs_grid[1:, :, :] - coeffs_grid[:-1, :, :]) / np.tile(coeffs_grid[[-1], :, :], (coeffs_grid.shape[0] - 1, 1, 1)) * 100
    MA_coeffs_grid_der_y = np.zeros(coeffs_grid_der_y.shape)
    for row in range(coeffs_grid_der_y.shape[0]):
        for column in range(coeffs_grid_der_y.shape[1]):
            for coeff in range(coeffs_grid_der_y.shape[2]):
                element = np.mean(coeffs_grid_der_y[row, max(0, column - horizon_length):column + 1, coeff])
                MA_coeffs_grid_der_y[row, column, coeff] = element

    # Moving average of the derivative along the number of data samples axis
    # Plot the cl coefficients
    coeff_plotter(MA_coeffs_grid_der_x[:, :, :degree_cla+1], degree_cla, "MA percentage change of the lift (q derivative)",
                  x_coords_mesh_der_x, y_coords_mesh_der_x, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$MA (D_q x_{{{c}}},10)\; [\\%]$",
                  switch_title=switch_title)

    # Plot the cd coeffs
    coeff_plotter(MA_coeffs_grid_der_x[:, :, degree_cla+1:], degree_cda, "MA percentage change of the drag (q derivative)",
                  x_coords_mesh_der_x, y_coords_mesh_der_x, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$MA (D_q y_{{{c}}},10)\; [\\%]$",
                  switch_title=switch_title)

    # Moving average of the derivative along the number of blade sections axis
    # Plot the cl coefficients
    coeff_plotter(MA_coeffs_grid_der_y[:, :, :degree_cla+1], degree_cla, "MA percentage change of the lift (n_bs derivative)",
                  x_coords_mesh_der_y, y_coords_mesh_der_y, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c: f"$MA (D_{{n_{{bs}}}} x_{{{c}}},10)\; [\\%]$",
                  switch_title=switch_title)

    # Plot the cd coeffs
    coeff_plotter(MA_coeffs_grid_der_y[:, :, degree_cla+1:], degree_cda, "MA percentage change of the drag (n_bs derivative)",
                  x_coords_mesh_der_y, y_coords_mesh_der_y, gradient=True, gradient_cb_min=-1, gradient_cb_max=1,
                  cbar_label_func=lambda c:f"$MA (D_{{n_{{bs}}}} y_{{{c}}},10)\; [\\%]$",
                  switch_title=switch_title)

    # Using the maximum
    # Plot the maximum values from the coefficients axis. As a result, the 6 plots that we have of the moving average
    # of the derivative wrt the number of samples is collapsed into a single plot. For each n_bs and q combination,
    # the MA of the derivative with the maximum value is stored. The same is done for the derivative wrt to the number
    # of blade sections. The first plot from the next two plots corresponds to Figure D.8 of the thesis.
    MA_coeffs_grid_der_x_max = np.reshape(np.amax(np.abs(MA_coeffs_grid_der_x), axis=2),
                                          [coeffs_grid_der_x.shape[0], coeffs_grid_der_x.shape[1], 1])
    MA_coeffs_grid_der_y_max = np.reshape(np.amax(np.abs(MA_coeffs_grid_der_y), axis=2),
                                          [coeffs_grid_der_y.shape[0], coeffs_grid_der_y.shape[1], 1])

    coeff_plotter(MA_coeffs_grid_der_x_max, 0, "percentage change of the maximum",
                  x_coords_mesh_der_x, y_coords_mesh_der_x, gradient=True,
                  cbar_label_func=lambda c: "$g_q(q, n_{bs})\; [\\%]$",
                  switch_title=switch_title)

    coeff_plotter(MA_coeffs_grid_der_y_max, 0, "percentage change of the maximum",
                  x_coords_mesh_der_y, y_coords_mesh_der_y, gradient=True,
                  cbar_label_func=lambda c: "$g_{n_{bs}}(q, n_{bs})\; [\\%]$",
                  switch_title=switch_title)

    # Computing the average along each of the axes. The previous plots are collapse along the n_bs or q axis. They are
    # used to create the Figures D.9 and D.10 in the thesis.
    g_qa = np.mean(np.reshape(MA_coeffs_grid_der_x_max,
                              [MA_coeffs_grid_der_x_max.shape[0], MA_coeffs_grid_der_x_max.shape[1]]), axis=0)
    g_nbs = np.mean(np.reshape(MA_coeffs_grid_der_y_max,
                               [MA_coeffs_grid_der_y_max.shape[0], MA_coeffs_grid_der_y_max.shape[1]]), axis=1)
    print(MA_coeffs_grid_der_x_max.shape, g_qa.shape)

    # Compute the plot collapsing the data along the q (number of samples) axis.
    global figure_number
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(x_coords[1:], g_qa, linewidth=4)
    plt.axhline(0.1, 0, x_coords[-1] + (x_coords[-1] - x_coords[-2]), color="black", alpha=0.6, linestyle="dashed",
                linewidth=4)
    ideal_point_g_qa_indeces = np.where(g_qa < threshold)[0]
    print(ideal_point_g_qa_indeces)
    if ideal_point_g_qa_indeces.shape[0] != 0:
        index = ideal_point_g_qa_indeces[0]
        plt.plot(x_coords[1+index], g_qa[index], "ro", markersize=10)
    plt.xlabel("q [-]")
    plt.ylabel("$h_q$(q) [%]")
    plt.grid(True)

    # Compute the plot collapsing the data along the n_bs (number of blade sections) axis.
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(y_coords[1:], g_nbs, linewidth=4)
    plt.axhline(0.1, 0, y_coords[-1] + (y_coords[-1] - y_coords[-2]), color="black", alpha=0.6, linestyle="dashed",
                linewidth=4)
    ideal_point_g_nbs_indeces = np.where(g_nbs < threshold)[0]
    if ideal_point_g_nbs_indeces.shape[0] != 0:
        index = ideal_point_g_nbs_indeces[0]
        plt.plot(y_coords[1+index], g_nbs[index], "ro", markersize=10)
    plt.xlabel("$n_{bs}$ [-]")
    plt.ylabel("$h_{n_{bs}}$($n_{bs}$) [%]")
    plt.grid(True)


def coeff_plotter(coeffs_grid_local, degree, coeff_type, X, Y, gradient=False, gradient_cb_min=0, gradient_cb_max=1,
                  cbar_label_func=lambda c:"", switch_title=True):
    """
    Used to create a colour map of the lift or drag coefficient
    :param coeffs_grid_local: identified cl and cd coefficients
    :param degree: degree of cl polynomial
    :param coeff_type: the type of coefficient, used for the plot title
    :param X: the values of the x-axis
    :param Y: the values of the y-axis
    :param gradient: whether the function is plotting a gradient
    :param gradient_cb_min: the lower bound for the colorbar
    :param gradient_cb_max: the higher bound for the colorbar
    :param cbar_label_func: the function to create the label for the colorbar
    :param switch_title: whether the title of the plot should be printed (not done with thesis/paper plots)
    :return: None
    """
    global figure_number
    fig = plt.figure(figure_number)
    figure_number += 1
    axes = fig.subplots(degree + 1, 1, gridspec_kw={'wspace': 0.5, 'hspace': 0.7})
    counter = 0
    colorbar_lst = []
    for ax in np.array([axes]).flatten():
        if gradient:
            im = ax.pcolormesh(X, Y, coeffs_grid_local[:, :, counter], vmin=gradient_cb_min, vmax=gradient_cb_max,
                               cmap="viridis")
        else:
            im = ax.pcolormesh(X, Y, coeffs_grid_local[:, :, counter], cmap="viridis")
        ax.set_xlabel("q [-]")
        ax.set_ylabel("$n_{bs}$ [-]")
        ax.yaxis.set_major_locator(MultipleLocator(200))
        ax.yaxis.set_minor_locator(IndexLocator(base=Y[1, 0]-Y[0, 0], offset=0))
        ax.xaxis.set_minor_locator(IndexLocator(base=X[0, 1]-X[0, 0], offset=0))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label=cbar_label_func(counter))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 1))
        cax.ticklabel_format(axis="y", style="sci", scilimits=(0, 1))
        colorbar_lst.append(cax)
        counter += 1
    fig.subplots_adjust(left=0.05, top=0.94, right=0.85, bottom=0.12)
    fig.set_size_inches(19.24, 10.55)
    if switch_title:
        fig.suptitle(f"Value of the {coeff_type} coefficients wrt. the number of samples and blade sections")


def adjust_plot(f, ax_lst):
    """
    Method that contains the heuristic to compute the required distance from the y-axes for the y-axis label given the
    length of the y-ticks. It also adjusts the size of the plot
    :param f: the figure object
    :param ax_lst: axis object
    :return: None
    """
    # Heuristic to compute the required distance from the y-axes for the y-axis label given the length of the y-ticks
    f_n_long_y_axis = 0
    for axs in ax_lst:
        f_n_long_y_axis += sum([j[j.index('.') + 1] != 'e' and j[j.index('.') + 1] != '0' and j[j.index('.') + 1] != '9'
                                for j in [np.format_float_scientific(i) for i in axs.get_yticks()]])

    for axs in ax_lst:
        if f_n_long_y_axis == 0:
            axs.yaxis.set_label_coords(-0.05, 0.5)
        else:
            axs.yaxis.set_label_coords(-0.095, 0.5)

    f.subplots_adjust(left=0.13, top=0.95, right=0.99, bottom=0.13)


def plot_FM(t, rotation_angle, F, M, mass_aero="m"):
    """
    Function to plot the changes in force and moments due to the failure of a blade. It is used for the creation of:
        - Figures 9.22, 9.23, 9.24, 9.25, 9.26, 9.27, 9.30, 9.31 in the thesis
        - Figs. 17, 18, 19, 20, 23, 24 of the paper: "Blade Element Theory Model for UAV Blade Damage Simulation"
    :param t: time vector
    :param rotation_angle: angle that the propeller has rotated
    :param F: 3D vector containing the forces
    :param M: 3D vector containing the moments
    :param mass_aero: the type of forces and moments that are fed to the function, whether they are mass related,
    aerodynamic related or something else
    :return: None
    """
    global figure_number

    # Plot of the propeller angle of rotation
    rotation_angle_deg = [degrees(i) for i in rotation_angle]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(t, rotation_angle_deg)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Evolution of propeller angle")
    plt.grid(True)

    # Plot of the forces and moments in two plots of 3 subplots each
    f_f, ax_f_lst = plt.subplots(3, 1, sharex=True, gridspec_kw={'wspace': 0.5,'hspace': 0.3})
    f_m, ax_m_lst = plt.subplots(3, 1, sharex=True, gridspec_kw={'wspace': 0.5,'hspace': 0.3})
    # f_f.suptitle("Evolution of Forces")  # commented out for the generation of paper plots
    # f_m.suptitle("Evolution of Moments")  # commented out for the generation of paper plots
    figure_number += 2
    axis_names = ["x", "y", "z"]
    for i in range(3):  # A subplot for each axis
        # FORCES
        ax_f = ax_f_lst[i]
        ax_f.plot(t, F[i, :], linewidth=3)

        if mass_aero == "a" or mass_aero == "m":
            ax_f.set_ylabel(f"$F^P_{{{mass_aero}_{axis_names[i]}}}$ [N]")
        else:
            ax_f.set_ylabel(f"$\Delta F^P_{{{axis_names[i]}}}$ [N]")
        ax_f.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
        ax_f.grid(True)

        # MOMENTS
        ax_m = ax_m_lst[i]
        ax_m.plot(t, M[i, :], linewidth=3)

        if mass_aero == "a" or mass_aero == "m":
            ax_m.set_ylabel(f"$M^P_{{{mass_aero}_{axis_names[i]}}}$ [Nm]")
        else:
            ax_m.set_ylabel(f"$\Delta M^P_{{{axis_names[i]}}}$ [Nm]")
        ax_m.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
        ax_m.grid(True)
    ax_f.set_xlabel("Time [s]")
    ax_m.set_xlabel("Time [s]")

    # Adjust the format of the figure
    adjust_plot(f_f, ax_f_lst)
    adjust_plot(f_m, ax_m_lst)
    plt.show()


def plot_FM_multiple(t, F, M, mass_aero="m", x_axis_label="Blade damage [%]"):
    """
    Function to plot the changes in force and moments due to the failure of a blade. In contrast with the previous
    function, multiple curves are plotted in the same graph. It is used for the creation of:
        - Figures 9.28, 9.29, 9.32, 9.33, 9.34, 9.35 in the thesis
        - Figs. 21, 22, 25, 26, 27, 28, 29, 30 of the paper: "Blade Element Theory Model for UAV Blade Damage Simulation"
    :param t: time vector
    :param F: 3D vector containing the forces
    :param M: 3D vector containing the moments
    :param mass_aero: the type of forces and moments that are fed to the function, whether they are mass related,
    aerodynamic related or something else
    :param x_axis_label: the label of the x axis
    :return: None
    """
    global figure_number
    figure_number += 2

    # Plot of the forces and moments in two plots of 3 subplots each
    f_f, ax_f_lst = plt.subplots(3, 1, sharex=True, gridspec_kw={'wspace': 0.5,'hspace': 0.3})
    f_m, ax_m_lst = plt.subplots(3, 1, sharex=True, gridspec_kw={'wspace': 0.5,'hspace': 0.3})
    # f_f.suptitle("Evolution of Forces")  # commented out for the generation of paper plots
    # f_m.suptitle("Evolution of Moments")  # commented out for the generation of paper plots
    axis_names = ["x", "y", "z"]  # A subplot for each axis
    number_curves = F.shape[2]
    for i in range(3):
        for j in range(number_curves):
            # FORCES
            ax_f = ax_f_lst[i]
            ax_f.plot(t, F[i, :, j], linewidth=3)

            if mass_aero == "a" or mass_aero == "m":  # when either mass or aerodynamic effects are plotted
                ax_f.set_ylabel(f"$F^P_{{{mass_aero}_{axis_names[i]}}}$ [N]")
            elif mass_aero == "b":  # when both, mass and aerodynamic effects are plotted together
                ax_f.set_ylabel(f"$F^P_{{._{axis_names[i]}}}$ [N]")
            else:
                ax_f.set_ylabel(f"$\Delta F^P_{{{axis_names[i]}}}$ [N]")
            ax_f.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
            ax_f.grid(True)
            f_f.tight_layout()

            # MOMENTS
            ax_m = ax_m_lst[i]
            ax_m.plot(t, M[i, :, j], linewidth=3)

            if mass_aero == "a" or mass_aero == "m":
                ax_m.set_ylabel(f"$M^P_{{{mass_aero}_{axis_names[i]}}}$ [Nm]")
            elif mass_aero == "b":
                ax_m.set_ylabel(f"$M^P_{{._{axis_names[i]}}}$ [Nm]")
            else:
                ax_m.set_ylabel(f"$\Delta M^P_{{{axis_names[i]}}}$ [Nm]")
            ax_m.ticklabel_format(axis="y", style="sci", scilimits=(-2, -7))
            ax_m.grid(True)
            f_m.tight_layout()

    # Adjusting plots and providing x-axis label
    ax_f.set_xlabel(x_axis_label)
    ax_m.set_xlabel(x_axis_label)

    # Adjust the format of the figure
    adjust_plot(f_f, ax_f_lst)
    adjust_plot(f_m, ax_m_lst)
    plt.show()


def plot_coeffs_params_blade_contribution(LS_terms, b):
    """
    Function that plots the contribution of each blade to the parameters used to identify the lift and drag coefficients
    In the paper "Blade Element Theory Model for UAV Blade Damage Simulation", it is going to plot the contribution of
    each blade to the terms in a row of the A matrix in Equation (46).
    Unfortunately, the author did not find great use for the information in this plot.
    :param LS_terms: values used for the Least Squares for each of the blades
    :param b: the b matrix with the thrust and torque information
    :return: None
    """
    n_blades = len(LS_terms)
    n_coeffs = LS_terms[0].shape[1]
    fig, ax = plt.subplots(2, 1)
    coefficient_labels = [f'C{i + 1}' for i in range(n_coeffs)]
    blades_labels = [f'Blade {i + 1}' for i in range(n_blades)]
    y_labels = ['Thrust', 'Torque']

    # One plot for thrust and another for torque
    for i in range(len(y_labels)):
        previous = np.zeros(n_coeffs)
        ax[i].axhline(y=b[i], color='r', linestyle='-')
        for j in range(n_blades):
            current = LS_terms[j][i, :]
            if j == 0:
                ax[i].bar(coefficient_labels, current, label=blades_labels[j])
            else:
                ax[i].bar(coefficient_labels, current, bottom=previous, label=blades_labels[j])
            previous += current
        ax[i].set_ylabel(f"{y_labels[i]}: Parameter value")
        ax[i].legend()
        ax[i].grid(True)

    fig.suptitle('Blade contribution to coefficient params')
    plt.show()


# multi_figure_storage("Saved_figures/500_dp_100_bs.pdf", figs=None, dpi=200)
def multi_figure_storage(filename, figs=None, dpi=200):
    """
    Function that creates a pdf with all the plots opened at the moment.
    The only gotcha here is that all figures are rendered as vector (pdf) graphics. If you want your figure to utilize
    raster graphics (i.e. if the files are too large as vectors), you could use the rasterized=True option when
    plotting quantities with many points. In that case the dpi option that I included might be useful.
    :param filename: name of the file where you want to save them, you need to include ".pdf"
    :param figs: whether you want to save a specific group of figures, then you insert the fig object
    :param dpi: the desired resolution
    :return:
    """
    pp = PdfPages(filename)
    if figs is None:  # In the case that no file names are provided, the open figures are retrieved
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:  # Save each of the found figures in pdf format
        fig.savefig(pp, dpi=dpi, format='pdf')
    pp.close()
