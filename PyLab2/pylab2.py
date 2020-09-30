import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

xdata, ydata = np.loadtxt("ohm data2.txt", unpack=True, delimiter="\t", skiprows=1)


accuracy_x = xdata * 0.0025  # DC Voltage Accuracy ±(0.25% of reading)
precision_x = 0.01  # Error of precision is 0.01V
sigma_x = np.maximum(accuracy_x, precision_x)

accuracy_y = ydata * 0.0075  # DC Current Accuracy ±(0.75% of reading)
precision_y = 0.1  # Error of precision is 0.1mA
sigma_y = np.maximum(accuracy_y, precision_y)


# Q1
def model_function(x, a, b):
    return a * x + b

p_opt, p_cov = curve_fit(model_function, xdata, ydata, (1, 0), sigma_y, True)

# plt.scatter(xdata,ydata, label="data")
fig, ax = plt.subplots()
ax.plot(xdata, model_function(xdata, *p_opt), 'r-',
        label='fit: a=%5.3f, b=%5.3f' % tuple(p_opt))
ax.errorbar(xdata, ydata, xerr=sigma_x, yerr=sigma_y, linestyle='None', label="error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('residuals')
ax.scatter(xdata, model_function(xdata, *p_opt) - ydata)
plt.show()

# Q2

def model_function2(x, a):
    return a * x

p_opt, p_cov = curve_fit(model_function2, xdata, ydata, (1), sigma_y, True)

# plt.scatter(xdata,ydata, label="data")
fig, ax = plt.subplots()
ax.plot(xdata, model_function2(xdata, *p_opt), 'r-',
        label='fit: a=%5.3f' % tuple(p_opt))
ax.errorbar(xdata, ydata, xerr=sigma_x, yerr=sigma_y, linestyle='None', label="error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('residuals')
ax.scatter(xdata, model_function2(xdata, *p_opt) - ydata)
plt.show()


# Q4

def chi2(xdata, ydata, measurement_error, model_function):
    p_opt, p_cov = curve_fit(model_function, xdata, ydata, (1, 0), measurement_error, True)
    n = len(p_opt)
    N = len(xdata)
    v = N - n
    expected = model_function(xdata, *p_opt)
    statistics = sum(((ydata - expected) / measurement_error) ** 2) / v
    return statistics


chi2(xdata, ydata, sigma_y, model_function)
