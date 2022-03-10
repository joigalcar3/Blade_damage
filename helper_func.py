import numpy as np
import matplotlib.pyplot as plt
from math import radians
from math import degrees
from time import time

figure_number = 1


# Helper functions
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
    area = (tc + bc) * h / 2
    return area


def compute_trapezoid_cg(bc, tc, h):
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
    chord = bc - (bc - tc) / h * (pos - h0)
    return chord


def compute_average_chord(chords, hs, pos_start, pos_end):
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
    def update_chords_h(counter, h_origin):
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


def compute_P52(x1, x2):
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
        new_time = time()
        elapsed_time = new_time - current_time
        current_time = new_time
        print(f'Iteration {i}. Elapsed time: {elapsed_time}')
    return current_time


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
        if not W_matrix:
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

    return x


# Plotters
def plot_chord_twist(chord, twist):
    global figure_number

    x_chord = range(len(chord))
    y_chord_1 = [i/2 for i in chord]
    y_chord_2 = [-i for i in y_chord_1]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(x_chord, y_chord_1, 'r-')
    plt.plot(x_chord, y_chord_2, 'r-')
    plt.grid(True)
    plt.show()

    x_twist = range(len(twist))
    y_twist = [degrees(i) for i in twist]
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(x_twist, y_twist, 'r-')
    plt.grid(True)
    plt.show()


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
        for i in range(degree_cla+1):
            cl += alpha ** i * x[i]
        return cl

    def cda_equation(alpha):
        """
        Creates the data points for the cd-alpha curve given an angle of attach and the cd-alpha coefficients
        :param alpha: angle of attack
        :return:
        """
        cd = 0
        for i in range(degree_cla+1, degree_cla + degree_cda + 2):
            cd += alpha ** i * x[i]
        return cd

    # Creation of the range of alphas to plot and the cl-cd coefficients
    alphas = np.arange(start_alpha, finish_alpha, 0.1)
    cls = [cla_equation(radians(aoa)) for aoa in alphas]
    cds = [cda_equation(radians(aoa)) for aoa in alphas]

    # Plot the cl-alpha curve
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(alphas, cls, 'r-')
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("Lift coefficient [-]")
    plt.title("Cl-alpha curve")
    plt.grid(True)

    # Plot the cd-alpha curve
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(alphas, cds, 'r-')
    plt.xlabel("Angle of attack [deg]")
    plt.ylabel("Drag coefficient [-]")
    plt.title("Cd-alpha curve")
    plt.grid(True)

    # Compare the results predicted by multiplying A and x, with respect to the observations in b for the thrust
    b_approx = np.matmul(A, x)
    number_p = A.shape[0]
    data_points = range(int(number_p/2))
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
    average_error_T = np.mean(error[::2])
    average_error_Q = np.mean(error[1::2])
    percentage_error = [error[i, 0]/b[i, 0] * 100 for i in range(error.shape[0])]
    print(f"Mean error thrust: {average_error_T} [N].")
    print(f"Mean error torque: {average_error_Q} [Nm].")
    print(f"Maximum error percentage: {max(percentage_error)}%")

    # Plot of error for thrust
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, error[::2], 'g-')
    plt.plot(data_points, np.repeat(average_error_T, len(data_points)), 'k--')
    plt.xlabel("Data point [-]")
    plt.ylabel("Approximation thrust error [N]")
    plt.title("Approximation error thrust")
    plt.grid(True)

    # Plot of error for torque
    plt.figure(figure_number)
    figure_number += 1
    plt.plot(data_points, error[1::2], 'g-')
    plt.plot(data_points, np.repeat(average_error_Q, len(data_points)), 'k--')
    plt.xlabel("Data point [-]")
    plt.ylabel("Approximation torque error [Nm]")
    plt.title("Approximation error torque")
    plt.grid(True)
    plt.show()

    # Plot of the angles of attack seen by each of the selected blade sections. It is represented as a box plot such
    # that it can be seen the average angle of attack as well as the range of alphas seen by its section.
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
        figure_number += 1
        plt.boxplot(np.degrees(aoa_vectors))
        plt.grid(True)
        plt.show()


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
    plt.show()


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
    coefficient_labels = [f'C{i+1}' for i in range(n_coeffs)]
    blades_labels = [f'Blade {i+1}' for i in range(n_blades)]
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
    plt.show()



