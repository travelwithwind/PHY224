import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

xdata, ydata = np.loadtxt(r"C:\Users\lenovo\Desktop\courses\PHY224\assignments\PyLab 4\data.txt", unpack=True, delimiter="\t", skiprows=1)


accuracy_x = xdata * 0.0025  # DC Voltage Accuracy ±(0.25% of reading)
precision_x = 0.01  # Error of precision is 0.01V
sigma_x = np.maximum(accuracy_x, precision_x)

accuracy_y = ydata * 0.0075  # DC Current Accuracy ±(0.75% of reading)
precision_y = 0.1  # Error of precision is 0.1mA
sigma_y = np.maximum(accuracy_y, precision_y)

# turn mA to A
ydata = ydata/1000
sigma_y = sigma_y/1000

# functions
def f(x, a, b):
    return a * x + b


def g(x, a, b):
    return a * np.power(x, b)


def theoretical(I_0, v):
    constant = I_0/(v[0]**0.5882)
    print(constant)
    return constant * (v ** 0.5882)



# linear regression on (log(xi), log(yi)) using f
logy = np.log(ydata)
logx = np.log(xdata)
sigma_logy = sigma_y / ydata
p_opt, p_cov = curve_fit(f, logx, logy, (1, 0), sigma_logy, True)

fig, ax = plt.subplots()
ax.loglog(xdata, np.exp(f(logx, *p_opt)), 'r-',
        label='fit: a=%5.3f, b=%5.3f' % tuple(p_opt))
ax.errorbar(xdata, ydata, xerr=sigma_x, yerr=sigma_y, label="error",marker=".")
ax.plot(xdata, theoretical(ydata[0], xdata), label="theoretical")
ax.set_title('Current(A) vs Voltage(V)')
ax.set_xlabel('Voltage(V)')
ax.set_ylabel('Current(A)')
ax.legend()
plt.show()
print("The estimate for blackbody values is {:.3f} with variance {:.3f}".format(p_opt[0], p_cov[0,0]) )


# nonlinear regression on (xi, yi) using g
p_opt, p_cov = curve_fit(g, xdata, ydata, (1, 0), sigma_y, True)
fig, ax = plt.subplots()
ax.loglog(xdata, g(xdata, *p_opt), 'r-',
        label='fit: a=%5.3f, b=%5.3f' % tuple(p_opt))
ax.errorbar(xdata, ydata, xerr=sigma_x, yerr=sigma_y, label="error",marker=".")
ax.plot(xdata, theoretical(ydata[0], xdata), label="theoretical")
ax.set_title('Current(A) vs Voltage(V)')
ax.set_xlabel('Voltage(V)')
ax.set_ylabel('Current(A)')
ax.legend()
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


chi2(logx, logy, sigma_logy, f)
chi2(xdata, ydata, sigma_y, g)

