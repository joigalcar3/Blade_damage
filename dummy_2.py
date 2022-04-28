import numpy as np
from scipy.optimize import minimize
from math import *
import time
import matplotlib.pyplot as plt

def personal_opt(func, x0, den):
    x = x0
    alpha = 1
    th = 0.01
    counter = 0
    previous_der = func(x)
    for i in range(10000):
        if abs(den([x])) < 1e-15:
            return x, i
        der = func(x)
        x_new = x - alpha*der
        if der != 0:
            if previous_der/der < 0:
                alpha = alpha/2.
                # print("Reduce alpha")

        step = x_new-x
        if step < th:
            counter += 1
        else:
            counter = 0

        x = x_new
        previous_der = der
        if counter > 50:
            return x, i

        # print(i, x, x_new, alpha, step, counter, der)

    return x, i

T = 2.9635930638797627
rho = 1.225
A = 0.017671458676442587
V = 2.6038555117916555
angle = 0.65742676

min_func = lambda x: abs(T - 2 * rho * A * x[0] * np.sqrt((V * np.cos(angle)) ** 2 + (V * np.sin(angle) + x[0]) ** 2))
min_func_extra = lambda Tt, Vt, alpha, x: abs(Tt - 2 * rho * A * x[0] * np.sqrt((Vt * np.cos(alpha)) ** 2 + (Vt * np.sin(alpha) + x[0]) ** 2))
min_func_2 = lambda x: T - 2 * rho * A * x[0] * np.sqrt((V * np.cos(angle)) ** 2 + (V * np.sin(angle) + x[0]) ** 2)
min_func_2_extra = lambda Tt, Vt, alpha, x: Tt - 2 * rho * A * x[0] * np.sqrt((Vt * np.cos(alpha)) ** 2 + (Vt * np.sin(alpha) + x[0]) ** 2)
x0 = np.array([1])
bnds = ((0, 20),)

# Uniform induced velocity and inflow
now_time = time.time()
v0 = minimize(min_func, x0, method='Nelder-Mead', tol=1e-6, options={'disp': False}, bounds=bnds).x[0]
then_time = time.time()
scipy_time = then_time-now_time

der_opt = lambda x: (-2*rho*A*np.sqrt((V*np.cos(angle))**2+(V*np.sin(angle)+x)**2) - \
                     2*rho*A*x*(V*np.sin(angle)+x)/(np.sqrt((V*np.cos(angle))**2+(V*np.sin(angle)+x)**2))) * \
                    min_func_2([x])/min_func([x])

der_opt_extra = lambda Vt, alpha, x: (-2*rho*A*np.sqrt((Vt*np.cos(alpha))**2+(Vt*np.sin(alpha)+x)**2) - \
                     2*rho*A*x*(Vt*np.sin(alpha)+x)/(np.sqrt((Vt*np.cos(alpha))**2+(Vt*np.sin(alpha)+x)**2)))

der_opt_extra_extra = lambda Tt, Vt, alpha, x: (-2*rho*A*np.sqrt((Vt*np.cos(alpha))**2+(Vt*np.sin(alpha)+x)**2) - \
                     2*rho*A*x*(Vt*np.sin(alpha)+x)/(np.sqrt((Vt*np.cos(alpha))**2+(Vt*np.sin(alpha)+x)**2))) * \
                    min_func_2_extra(Tt, Vt, alpha, [x])/min_func_extra(Tt, Vt, alpha, [x])

now_time = time.time()
v0_2 = personal_opt(der_opt, x0, min_func)
then_time = time.time()
personal_time = then_time-now_time
print("scipy_time = ", scipy_time)
print("personal_time = ", personal_time)

x_range = np.arange(-10, 20, 0.1)

plt.figure(5001)
plt.plot(x_range, [der_opt_extra_extra(0.19, 4, radians(-70), i) for i in x_range])
plt.plot(x_range, [der_opt_extra_extra(0.19, 4, radians(-75), i) for i in x_range])
plt.plot(x_range, [der_opt_extra(4, radians(-70), i) for i in x_range])
plt.plot(x_range, [der_opt_extra(4, radians(-75), i) for i in x_range])
plt.grid(True)

plt.figure(5000)
plt.plot(x_range, [min_func_extra(0.19, 4, radians(-70), [i]) for i in x_range])
plt.plot(x_range, [min_func_2_extra(0.19, 4, radians(-70), [i]) for i in x_range])
plt.plot(x_range, [min_func_extra(0.19, 4, radians(-75), [i]) for i in x_range])
plt.grid(True)

print("hola")
#7.411585235595717
#plt.plot(alpha, [min_func_2([i]) for i in alpha])