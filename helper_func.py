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
from math import radians
from math import degrees
import time
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')

__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Development"

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
    :return:
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
            c1 = compute_chord_trapezoid(bc, tc, h, h0, pos_start)
            if (h0 + h) <= pos_end:
                area += compute_trapezoid_area(c1, tc, h - pos_start)
                c1 = tc
                pos_start = h0 + h

            if (h0 + h) > pos_end >= h0:
                c2 = compute_chord_trapezoid(bc, tc, h, h0, pos_end)
                area += compute_trapezoid_area(c1, c2, pos_end - pos_start)
        h0 += h

    average_chord = area / distance
    return average_chord


def compute_average_chords(chords, hs, n_segments):
    """
    Function that computes the average chords of all the blade section
    :param chords: list with all the blade sections' chords
    :param hs: list with all the blade sections' lengths
    :param n_segments: number of blade sections
    :return:
    """

    def update_chords_h(counter, h_origin):
        """
        Function that retrieves the information related to a blade section
        :param counter: current blade section index
        :param h_origin: start location of the blade section along the blade axis
        :return:
        """
        if counter >= 0:
            h_origin += hs[counter]
        counter += 1
        current_h = hs[counter]
        current_bc = chords[counter]
        current_tc = chords[counter + 1]
        return current_h, current_bc, current_tc, counter, h_origin

    length_blade = sum(hs)
    length_segment = length_blade / n_segments

    segment = 0
    h, bc, tc, trapezoid_count, h0 = update_chords_h(-1, 0)
    average_chords = []
    segment_chords = [bc]
    for i in range(n_segments):
        area_1 = 0
        pos_c1 = segment * length_segment
        pos_c2 = (segment + 1) * length_segment
        c1 = segment_chords[segment]
        if pos_c2 > (h0 + hs[trapezoid_count]):  # Transition is incorrect
            area_1 = compute_trapezoid_area(c1, tc, h0 + h - pos_c1)
            h, bc, tc, trapezoid_count, h0 = update_chords_h(trapezoid_count, h0)
        c2 = compute_chord_trapezoid(bc, tc, h, h0, pos_c2)

        if not bool(area_1):
            area = compute_trapezoid_area(c1, c2, length_segment)
        else:
            area = area_1 + compute_trapezoid_area(bc, c2, pos_c2 - h0)
        average_chord = area / length_segment
        average_chords.append(average_chord)
        segment_chords.append(c2)
        segment += 1
    return average_chords, segment_chords


# Helper functions related to aerodynamic Matlab model
# %%
def compute_P52(x1, x2):
    """
    Function from the Matlab Bebop model for computing a 5th degree polynomial with two variables
    :param x1: first variable
    :param x2: second variable
    :return:
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
    :return:
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
    Function to compute the psi value from the aerodynamic Matlab model
    :param beta: sideslip angle
    :param arm_angle: angle of the rotor arm with the body
    :param propeller_number: number of the propeller
    :return:
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
    Function obtained from the Matlab Bebop model to compute a 3rd order polynomial with two variables
    :param x1: first variable
    :param x2: second variable
    :param U: additional input
    :return:
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
    Function retrieved from the Matlab Bebop model that computes the nth order Fourier series
    :param x: variable for obtaining the Fourier series
    :param n: degree of the Fourier series
    :param U: additional input
    :param alpha: horizontal advance ratio
    :param beta: sideslip angle equation
    :return:
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
    :return:
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
            arguments_constraint = (kwargs["degree_cla"], kwargs["degree_cda"], kwargs["min_angle"], kwargs["max_angle"])
            # noinspection SpellCheckingInspection
            constraints = ({"type": "ineq", "fun": nonlinear_constraint_drag_minimum, "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_maximum, "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_decaying, "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_positive_slope,
                            "args": arguments_constraint},
                           {"type": "ineq", "fun": nonlinear_constraint_lift_alpha0, "args": arguments_constraint})
            x = minimize(minimize_func, x0, args=(A, b), method=min_method, constraints=constraints,
                         options={"disp": True, "maxiter": 2000}).x
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
    Constraint to avoid that the lift value to obtain a very large value, namely higher than 3
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
    :return:
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
                            activate_plotting=False, input_storage=None, warm_starts=None):
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
    :return:
    """
    global figure_number

    # Creating the row for the coefficients' grid
    coeffs_grid_row = np.zeros((1, len(number_samples_lst), degree_cla+degree_cda+2))
    input_storage_local = {'ylabel': input_storage['ylabel'], 'title': input_storage['title']}
    for i, number_samples in enumerate(number_samples_lst):
        # Retrieving the information concerning the first "number_samples" data points
        A_local = A[:number_samples*2, :]
        b_local = b[:number_samples*2, :]

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


# Plotters
# %%
def plot_chord_twist(chord, twist):
    """
    Two plots:
        - The first plot shows the chord of the blade
        - The second plot shows the twist of the blade along its span
    :param chord: list of chord values
    :param twist: list of twist values
    :return:
    """
    global figure_number

    x_chord = range(len(chord))
    y_chord_1 = [i / 2 for i in chord]
    y_chord_2 = [-i for i in y_chord_1]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(x_chord, y_chord_1, 'r-')
    plt.plot(x_chord, y_chord_2, 'r-')
    plt.grid(True)

    x_twist = range(len(twist))
    y_twist = [degrees(i) for i in twist]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(x_twist, y_twist, 'r-')
    plt.grid(True)


def plot_cla(x, A, b, aoa_storage, start_alpha, finish_alpha, degree_cla, degree_cda):
    """
    Function that plots the cl-alpha and cd-alpha curves. It also plots the average angle of attack seen by each blade
    element in the form of a box plot
    :param x: vector of unknown states
    :param A: regression matrix
    :param b: observation vector
    :param aoa_storage:
    :param start_alpha: first angle of attack to plot
    :param finish_alpha: last angle of attack to plot
    :param degree_cla: degree of the cl-alpha polynomial
    :param degree_cda: degree of the cd-alpha polynomial
    :return:
    """
    global figure_number

    def cla_equation(alpha):
        """
        Creates the data points for the cl-alpha curve given an angle of attach and the cl-alpha coefficients
        :param alpha: angle of attack
        :return:
        """
        cl = 0
        for i in range(degree_cla + 1):
            cl += alpha ** i * x[i]
        return cl

    def cda_equation(alpha):
        """
        Creates the data points for the cd-alpha curve given an angle of attach and the cd-alpha coefficients
        :param alpha: angle of attack
        :return:
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

    # Plot the cl-alpha curve
    title = "Cl-alpha curve: Cl = "
    for i in range(degree_cla + 1):
        title += f'{np.round(x[i].item(),2)} $\\alpha^{i}$'
        if i != degree_cla:
            if x[i+1].item() > 0:
                title += '+'
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(alphas, cls, 'r-')
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("Lift coefficient [-]")
    plt.title(title)
    ax = plt.gca()
    if (max(cls) - min(cls)) / 0.1 > 20:
        y_discretisation = 0.5
    elif (max(cls) - min(cls)) / 0.1 > 2:
        y_discretisation = 0.1
    else:
        y_discretisation = 0.01
    ax.yaxis.set_major_locator(MultipleLocator(y_discretisation))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.grid(True)

    # Plot the cd-alpha curve
    title = "Cd-alpha curve: Cd = "
    for i in range(degree_cla + 1, degree_cla + degree_cda + 2):
        title += f'{np.round(x[i].item(),2)} $\\alpha^{i-(degree_cla + 1)}$'
        if i != degree_cla + degree_cda + 1:
            if x[i+1].item() > 0:
                title += '+'
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(alphas, cds, 'r-')
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("Drag coefficient [-]")
    plt.title(title)
    ax = plt.gca()
    if (max(cds) - min(cds)) / 0.1 > 20:
        y_discretisation = 0.5
    elif (max(cds) - min(cds)) / 0.1 > 2:
        y_discretisation = 0.1
    else:
        y_discretisation = 0.01
    ax.yaxis.set_major_locator(MultipleLocator(y_discretisation))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.grid(True)

    # Compare the results predicted by multiplying A and x, with respect to the observations in b for the thrust
    b_approx = np.matmul(A, x)
    number_p = A.shape[0]
    data_points = range(int(number_p / 2))
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, b[::2], 'ro')
    plt.plot(data_points, b_approx[::2], 'bo')
    plt.xlabel("Data point [-]")
    plt.ylabel("Thrust value")
    plt.title("Thrust: Ax vs b")
    plt.grid(True)

    # Compare the results predicted by multiplying A and x, with respect to the observations in b for the torque
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, b[1::2, 0], 'ro')
    plt.plot(data_points, b_approx[1::2, 0], 'bo')
    plt.xlabel("Data point [-]")
    plt.ylabel("Torque value")
    plt.title("Torque: Ax vs b")
    plt.grid(True)

    # Computation of error
    error = b - b_approx
    error_T = error[::2]
    error_Q = error[1::2]
    average_error_T = np.mean(error_T)
    average_error_Q = np.mean(error_Q)
    percentage_error = [error[i, 0] / b[i, 0] * 100 for i in range(error.shape[0])]
    print(f"Mean error thrust: {average_error_T} [N].")
    print(f"Mean error torque: {average_error_Q} [Nm].")
    print(f"Maximum error percentage: {max(percentage_error)}%")

    # Computation of relative or normalized RMSE
    # Divide RMSE by difference between max and min of observed values
    RMSE_T1 = np.sqrt(np.mean(np.power(error_T, 2))) / (np.max(b[::2]) - np.min(b[::2]))
    RMSE_Q1 = np.sqrt(np.mean(np.power(error_Q, 2))) / (np.max(b[1::2]) - np.min(b[1::2]))

    # Divide RMSE by standard deviation of observed values
    RMSE_T2 = np.sqrt(np.mean(np.power(error_T, 2))) / np.std(b[::2])
    RMSE_Q2 = np.sqrt(np.mean(np.power(error_Q, 2))) / np.std(b[1::2])

    # Plot of error for thrust
    text_T_RMSE = f'Diff RMSE = {np.round(RMSE_T1, 2)} and Std RMSE = {np.round(RMSE_T2, 2)}'
    text_T_mean = f'Mean = {np.round(average_error_T, 4)}'
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, error[::2], 'g-', label=text_T_RMSE)
    plt.plot(data_points, np.repeat(average_error_T, len(data_points)), 'k--', label=text_T_mean)
    plt.xlabel("Data point [-]")
    plt.ylabel("Approximation thrust error [N]")
    plt.title("Approximation error thrust")
    plt.legend()
    plt.grid(True)

    # Plot of error for torque
    text_Q_RMSE = f'Diff RMSE = {np.round(RMSE_Q1, 2)} and Std RMSE = {np.round(RMSE_Q2, 2)}'
    text_Q_mean = f'Mean = {np.round(average_error_Q, 4)}'
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, error[1::2], 'g-', label=text_Q_RMSE)
    plt.plot(data_points, np.repeat(average_error_Q, len(data_points)), 'k--', label=text_Q_mean)
    plt.xlabel("Data point [-]")
    plt.ylabel("Approximation torque error [Nm]")
    plt.title("Approximation error torque")
    plt.grid(True)
    plt.legend()

    ## Validation plots to observe the whiteness of the residual. This function was commented out since the acorr
    ## function from matplotlib is able to do exactly the same
    # n_shifts = error_T.shape[0] - 1
    # lst_T = np.ones(2 * n_shifts - 1)
    # lst_Q = np.ones(2 * n_shifts - 1)
    # lst_T_center = np.matmul(error_T.T, error_T)
    # lst_Q_center = np.matmul(error_Q.T, error_Q)
    # lst_T[n_shifts] = 1
    # lst_Q[n_shifts] = 1
    # for i in range(1, n_shifts):
    #     value_T = np.matmul(error_T[:-i, :].T, error_T[i:, :])
    #     value_Q = np.matmul(error_Q[:-i, :].T, error_Q[i:, :])
    #     lst_T[n_shifts - 1 + i] = value_T / lst_T_center
    #     lst_T[n_shifts - 1 - i] = value_T / lst_T_center
    #     lst_Q[n_shifts - 1 + i] = value_Q / lst_Q_center
    #     lst_Q[n_shifts - 1 - i] = value_Q / lst_Q_center

    ## Plot corresponding to the thrust
    # x_axis = list(range(-n_shifts + 1, n_shifts))
    # conf = 1.96 / np.sqrt(error_T.shape[0])
    # plt.figure(figure_number)
    # figure_number += 1
    # plt.plot(x_axis, lst_T, 'b-')
    # plt.axhline(y=conf, xmin=-error_T.shape[0], xmax=error_T.shape[0], color="red", linestyle="--")
    # plt.axhline(y=-conf, xmin=-error_T.shape[0], xmax=error_T.shape[0], color="red", linestyle="--")
    # plt.xlabel("Number of lags")
    # plt.ylabel("Error autocorrelation")
    # plt.title("Autocorrelation of model residual: Thrust")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    conf = 1.96 / np.sqrt(error_T.shape[0])
    plt.figure(figure_number)
    ax = plt.gca()
    figure_number += 1
    ax.acorr(np.reshape(error_T, [-1, ]), maxlags=None, usevlines=False, normed=True, linestyle="-", marker='',
             color="b")
    plt.axhline(y=conf, xmin=-error_T.shape[0], xmax=error_T.shape[0], color="red", linestyle="--")
    plt.axhline(y=-conf, xmin=-error_T.shape[0], xmax=error_T.shape[0], color="red", linestyle="--")
    plt.xlabel("Number of lags")
    plt.ylabel("Error autocorrelation")
    plt.title("Autocorrelation of model residual: Thrust")
    plt.grid(True)
    plt.legend()

    ## Plot corresponding to the torque
    # plt.figure(figure_number)
    # figure_number += 1
    # plt.plot(x_axis, lst_Q, 'b-')
    # plt.axhline(y=conf, xmin=-error_Q.shape[0], xmax=error_Q.shape[0], color="red", linestyle="--")
    # plt.axhline(y=-conf, xmin=-error_Q.shape[0], xmax=error_Q.shape[0], color="red", linestyle="--")
    # plt.xlabel("Number of lags")
    # plt.ylabel("Error autocorrelation")
    # plt.title("Autocorrelation of model residual: Torque")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    plt.figure(figure_number)
    ax = plt.gca()
    figure_number += 1
    ax.acorr(np.reshape(error_Q, [-1, ]), maxlags=None, usevlines=False, normed=True, linestyle="-", marker='',
             color="blue")
    plt.axhline(y=conf, xmin=-error_Q.shape[0], xmax=error_Q.shape[0], color="red", linestyle="--")
    plt.axhline(y=-conf, xmin=-error_Q.shape[0], xmax=error_Q.shape[0], color="red", linestyle="--")
    plt.xlabel("Number of lags")
    plt.ylabel("Error autocorrelation")
    plt.title("Autocorrelation of model residual: Torque")
    plt.grid(True)
    plt.legend()

    # Plot of the angles of attack seen by each of the selected blade sections. It is represented as a box plot such
    # that it can be seen the average angle of attack as well as the range of alphas seen by its section.
    if aoa_storage != None:
        n_blade_sections = len(aoa_storage.keys())
        n_aoa = len(aoa_storage[0])
        flag = True
        while flag:
            user_input = input(f'Out of {n_blade_sections} blade sections, which ones would you like to plot? '
                               f'Please give the start and end sections separated by a comma.')

            # If the user uses the letter s, it means that it wants to stop without plotting.
            if user_input == "s":
                flag = False
                continue

            # Check that feasible blade sections have been given
            user_input = list(map(int, user_input.split(',')))
            start_section, end_section = user_input
            start_section, end_section = [0, n_blade_sections-1]
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

            plt.figure(figure_number)
            ax = plt.gca()
            figure_number += 1
            plt.boxplot(np.degrees(aoa_vectors), showfliers=False, patch_artist=True)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            plt.ylabel(r"$\alpha$ [deg]")
            plt.xlabel("Blade section number [-]")
            plt.grid(True, alpha=0.5)


def plot_inputs(inputs_dict):
    """
    Function to plot the inputs used for the data point generation
    :param inputs_dict: dictionary with the value of the inputs used
    :return:
    """
    global figure_number
    inputs = inputs_dict.keys()
    for key in inputs:
        if key != "ylabel" and key != "title":
            if type(inputs_dict[key][0]) != float:
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
            else:
                plt.figure(figure_number)
                figure_number += 1
                plt.plot(inputs_dict[key], "bo")
                plt.xlabel("Data point number [-]")
                plt.ylabel(inputs_dict["ylabel"][0][key])
                plt.title(inputs_dict["title"][0][key])
                plt.grid(True)
                # plt.show()


def plot_coeffs_map(coeffs_grid, degree_cla, degree_cda):
    """
    Function to plot the coefficients for different blade discretisations and number of data points
    :param coeffs_grid: identified cl and cd coefficients
    :param degree_cla: degree of cl polynomial
    :param degree_cda: degree of cd polynomial
    :return:
    """
    global figure_number

    def coeff_plotter(coeffs_grid_local, degree, coeff_type):
        global figure_number
        fig = plt.figure(figure_number)
        figure_number += 1
        axes = fig.subplots(degree + 1, 1, gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
        counter = 0
        for ax in axes:
            im = ax.pcolormesh(coeffs_grid_local[:, :, counter], cmap="viridis")
            ax.set_xlabel("q [-]")
            ax.set_ylabel("$n_{bs}$ [-]")
            ax.grid(True)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            counter += 1
        fig.suptitle(f"Value of the {coeff_type} coefficients wrt. the number of samples and blade sections")

    # Plot the cl coefficients
    coeff_plotter(coeffs_grid[:, :, :degree_cla+1], degree_cla, "lift")

    # Plot the cd coeffs
    coeff_plotter(coeffs_grid[:, :, degree_cla+1:], degree_cda, "drag")


def plot_FM(t, rotation_angle, F, M):
    """
    Method to plot the changes in force and moments due to the failure of a blade
    :param t: time vector
    :param rotation_angle: angle that the propeller has rotated
    :param F: 3D vector containing the forces
    :param M: 3D vector containing the moments
    :return:
    """
    global figure_number

    rotation_angle_deg = [degrees(i) for i in rotation_angle]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(t, rotation_angle_deg)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Evolution of propeller angle")
    plt.grid(True)

    f_f, ax_f_lst = plt.subplots(3, 1)
    f_f.suptitle("Evolution of Forces")
    f_m, ax_m_lst = plt.subplots(3, 1)
    f_m.suptitle("Evolution of Moments")
    figure_number += 2
    for i in range(3):
        ax_f = ax_f_lst[i]
        ax_f.plot(t, F[i, :])
        ax_f.set_xlabel("Time [s]")
        ax_f.set_ylabel("Force [N]")
        ax_f.grid(True)
        f_f.tight_layout()

        ax_m = ax_m_lst[i]
        ax_m.plot(t, M[i, :])
        ax_m.set_xlabel("Time [s]")
        ax_m.set_ylabel("Moment [Nm]")
        ax_m.grid(True)
        f_m.tight_layout()
    # plt.show()


def plot_coeffs_params_blade_contribution(LS_terms, b):
    """
    Function that plots the contribution of each blade to the parameters used to identify the lift and drag coefficients
    :param LS_terms: values used for the Least Squares for each of the blades
    :param b: the b matrix with the thrust and torque information
    :return:
    """
    n_blades = len(LS_terms)
    n_coeffs = LS_terms[0].shape[1]
    fig, ax = plt.subplots(2, 1)
    coefficient_labels = [f'C{i + 1}' for i in range(n_coeffs)]
    blades_labels = [f'Blade {i + 1}' for i in range(n_blades)]
    y_labels = ['Thrust', 'Torque']

    for i in range(2):
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
    # plt.show()


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
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, dpi=dpi, format='pdf')
    pp.close()
