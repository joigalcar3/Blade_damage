# Modules to import
from helper_func import *
from user_input import *
from Propeller import Propeller
from helper_func import *
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

a1 = np.load("Saved_figures/500_16000_dp_50_250_bs_3_va_SLSQP_mod/Data/500_16000_dp_50_250_bs_3_va_SLSQP_mod__coeffs_storage.npy")
a2 = np.load("Saved_figures/500_16000_dp_300_450_bs_3_va_SLSQP_mod/Data/500_16000_dp_300_450_bs_3_va_SLSQP_mod__coeffs_storage.npy")
a3 = np.load("Saved_data/500_16000_dp_500_550_bs_3_va_SLSQP_mod__coeffs_storage.npy")

k4 = np.load("Saved_data/500_16450_dp_50_550_bs_3_va_SLSQP_mod__coeffs_storage.npy")
data = np.vstack((a1,a2,a3))
y_coords = list(np.arange(50, 350, 50))               # [-] number of sections in which a single blade is divided
x_coords = list(np.arange(500, 16500, 50))

degree_cla = 2
degree_cda = 2

plot_coeffs_map(data, degree_cla, degree_cda, x_coords, y_coords)
plot_derivative_coeffs_map(data, degree_cla, degree_cda, x_coords, y_coords)



print("hola")