import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


# functions
def rc(x, r, c):
    return np.exp(-x / (r * c))


def rc_(x, tao):
    return np.exp(-x / tao)


def lr(x, R, L):
    return np.exp(-x * R / L)


# curve fitting on DC RC circuit
data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\dc 100kohm.csv", delimiter=',', skiprows=1)

# choose a segment of the data that contains one decay
data_seg = data[531:755]
xdata = data_seg['second'] - data_seg['second'].iloc[0]  # set the beginning of timeline to 0
ydata = data_seg['Volt'] / data_seg['Volt'].iloc[0]  # set it as a ratio over the initial value

plt.plot(xdata, ydata)
plt.xlabel("t")
plt.ylabel("V(t)/V0")

# accuracy_y = ydata * 0.0025 # Voltage Accuracy ±(0.25% of reading)
# precision_y = 0.000001 # Error of precision
# sigma_y = np.maximum(accuracy_y, precision_y)
sigma_y = np.ones_like(xdata) * 0.000001
sigma_x = np.ones_like(xdata) * 0.000001  # the error is chosen to be the "chop-off" precision error

p_opt, p_cov = curve_fit(rc, xdata, ydata, (100e3, 1e-6), sigma_y, True)

fig, ax = plt.subplots()
ax.plot(xdata, rc(xdata, *p_opt), 'r-',
        label='fit: R=%.3e, C=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error", linestyle='None', marker=".")
ax.set_title('RC circuit with battery')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Voltage(ratio)')
ax.legend()
plt.show()
time_constant = p_opt[0] * p_opt[1]
time_constant_sd = time_constant * np.sqrt((p_cov[0, 0] / p_opt[0]) ** 2 + (p_cov[1, 1] / p_opt[1]) ** 2)
print("The time constant estimated is {:.3e} with standard deviation {:.3e}".format(time_constant, time_constant_sd))

# curve fitting on AC RC circuit


data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\square rc 3kR.csv", delimiter=',',
                   skiprows=1)
data.plot(x='second')
# choose a segment of the data that contains one decay
data_seg = data[200:400]
xdata = data_seg['second'] - data_seg['second'].iloc[0]  # set the beginning of timeline to 0
ydata = data_seg['Volt.1'] / data_seg['Volt.1'].iloc[0]  # set it as a ratio over the initial value
plt.plot(xdata, ydata)

# accuracy_y = ydata * 0.00025 # Voltage Accuracy ±(2% of reading)
# precision_y = 0.000001 # Error of precision
# sigma_y = np.maximum(accuracy_y, precision_y)
sigma_y = np.ones_like(ydata) * 0.000001  # the error is chosen to be the "chop-off" precision error
sigma_x = np.ones_like(xdata) * 0.000001  # the error is chosen to be the "chop-off" precision error

p_opt, p_cov = curve_fit(rc, xdata, ydata, (1e5, 2.2e-8), sigma_y, True)

fig, ax = plt.subplots()
ax.plot(xdata, rc(xdata, *p_opt), 'r-',
        label='fit: R=%.3e, C=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata,  yerr=sigma_y, label="error",linestyle='None', marker=".")
ax.set_title('RC circuit')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Voltage(ratio)')
ax.legend()
plt.show()
time_constant = p_opt[0] * p_opt[1]
time_constant_sd = time_constant * np.sqrt((p_cov[0, 0] / p_opt[0]) ** 2 + (p_cov[1, 1] / p_opt[1]) ** 2)
print("The time constant estimated is {:.3e} with standard deviation {:.3e}".format(time_constant, time_constant_sd))

# curve fitting on LR circuit

data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\square rl 3kR 390khz.csv", delimiter=',',
                   skiprows=1)
data.plot(x='second')
# choose a segment of the data that contains one decay
data_seg = data[499:741]
# data_seg = data[229:484]
xdata = data_seg['second'] - data_seg['second'].iloc[0]  # set the beginning of timeline to 0
ydata = data_seg['Volt.1'] / data_seg['Volt.1'].iloc[1]  # set it as a ratio over the initial value
plt.plot(xdata, ydata)

sigma_y = np.ones_like(ydata) * 0.000001  # the error is chosen to be the "chop-off" precision error
sigma_x = np.ones_like(xdata) * 0.000001  # the error is chosen to be the "chop-off" precision error

p_opt, p_cov = curve_fit(lr, xdata, ydata, (1, 0.0426), sigma_y, True)

fig, ax = plt.subplots()
ax.plot(xdata, lr(xdata, *p_opt), 'r-',
        label='fit: R=%.3e, L=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error",linestyle='None', marker=".")
ax.set_title('RL circuit')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Voltage(ratio)')
ax.legend()
plt.show()
time_constant = p_opt[1] / p_opt[0]
time_constant_sd = time_constant * np.sqrt((p_cov[0, 0] / p_opt[0]) ** 2 + (p_cov[1, 1] / p_opt[1]) ** 2)
print("The time constant estimated is {:.3e} with standard deviation {:.3e}".format(time_constant, time_constant_sd))

# curve fitting on LC circuit with VL measured

# Theoretically, frequency is
# 1/(2*np.pi*np.sqrt(22e-9*42.6e-3))
# 5198.810551134781

data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\lc vc measured.csv", delimiter=',',
                   skiprows=1)
data.plot(x='second')
# observe 10 cycles in 0.002 seconds, so f=5000

data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\lc vl measured.csv", delimiter=',',
                   skiprows=1)
data.plot(x='second')
# observe 10 cycles in 0.002 seconds, so f=5000

################# experiment 2 #############

# impedance RC

# theoretical curve
R = 512.4
C = 22e-9
L = 42.6e-3
freq = np.linspace(10, 1300, 100)
angular_freq = 2 * np.pi * freq
Z_RC = np.sqrt(R ** 2 + (1 / (angular_freq * C)) ** 2)
Z_RL = np.sqrt(R ** 2 + (angular_freq * L) ** 2)
Z_RCL = np.sqrt(R ** 2 + (angular_freq * L - 1 / (angular_freq * C)) ** 2)
plt.plot(angular_freq, Z_RC)
plt.plot(angular_freq, Z_RL)
plt.plot(angular_freq, Z_RCL)

data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\impedance rc.csv", delimiter=',',
                   skiprows=1)
data['angular_freq'] = 2 * np.pi * data['freq']
data['Z_RC'] = data.V1 / data.V2 * R + R
fig, ax = plt.subplots()
ax.plot(data['angular_freq'], data['Z_RC'], label="observed")
R = 10000
C = 22e-9
freq = np.linspace(100, 4100, 100)
angular_freq = 2 * np.pi * freq
Z_RC = np.sqrt(R ** 2 + (1 / (angular_freq * C)) ** 2)
ax.plot(angular_freq, Z_RC, label="theoretical")
ax.legend()
ax.set_xlabel('Angular Frequency')
ax.set_ylabel('Z')
plt.show()

sub_data = data[:]
def impedance_rc(angular_freq, C):
    R = 10000
    Z_RC = np.sqrt(R ** 2 + (1 / (angular_freq * C)) ** 2)
    return Z_RC

xdata = sub_data['angular_freq']
ydata = sub_data['Z_RC']
reading_error = 0.03  # the error is chosen to be the "chop-off" precision error
sigma_x = np.ones_like(xdata) * 0.000001  # the error is chosen to be the "chop-off" precision error
sigma_y = ydata * np.sqrt((sub_data['V1']*reading_error / sub_data['V1']) ** 2 + (sub_data['V2']*reading_error / sub_data['V2']) ** 2)

p_opt, p_cov = curve_fit(impedance_rc, xdata, ydata, (22e-9), sigma_y, True)

fig, ax = plt.subplots()
ax.plot(xdata, impedance_rc(xdata, *p_opt), 'r-',
        label='fit: C=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error",linestyle='None', marker=".")
ax.set_title('RC circuit')
ax.set_xlabel('freqency')
ax.set_ylabel('Z')
ax.legend()
plt.show()
print("The capacitance estimated is {:.3e} with standard deviation {:.3e}".format(p_opt[0], p_cov[0,0]))

# impedance RL

# theoretical curve
R = 512.4
L = 42.6e-3
freq = np.linspace(1, 50000, 100)
angular_freq = 2 * np.pi * freq
Z_RL = np.sqrt(R ** 2 + (angular_freq * L) ** 2)
Z_RCL = np.sqrt(R ** 2 + (angular_freq * L - 1 / (angular_freq * C)) ** 2)

data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\impedance RL.csv", delimiter=',',
                   skiprows=1)
data['freq'] = data['freq'] * 1000
data['angular_freq'] = 2 * np.pi * data['freq']
data['V1'] = data['V1']/1000
data['V2'] = data['V2']/1000
data['Z_RL'] = data.V1 / data.V2 * R
fig, ax = plt.subplots()
ax.plot(data['angular_freq'], data['Z_RL'], label="observed")
ax.plot(angular_freq, Z_RCL, label="theoretical RCL")
ax.plot(angular_freq, Z_RL, label="theoretical")
ax.legend()
ax.set_xlabel('Angular Frequency')
ax.set_ylabel('Z')
plt.show()

sub_data = data[0:6]
def impedance_rl(angular_freq, L):
    R = 512.4
    Z_RL = np.sqrt(R ** 2 + ( (angular_freq * L)) ** 2)
    return Z_RL

xdata = sub_data['angular_freq']
ydata = sub_data['Z_RL']
reading_error = 0.03  # the error is chosen to be the "chop-off" precision error
sigma_x = np.ones_like(xdata) * 0.000001  # the error is chosen to be the "chop-off" precision error
sigma_y = ydata * np.sqrt((sub_data['V1']*reading_error / sub_data['V1']) ** 2 + (sub_data['V2']*reading_error / sub_data['V2']) ** 2)

p_opt, p_cov = curve_fit(impedance_rl, xdata, ydata, (0.04), sigma_y, True)

fig, ax = plt.subplots()
ax.plot(xdata, impedance_rl(xdata, *p_opt), 'r-',
        label='fit: C=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error",linestyle='None', marker=".")
ax.set_title('RC circuit')
ax.set_xlabel('freqency')
ax.set_ylabel('Z')
ax.legend()
plt.show()
print("The capacitance estimated is {:.3e} with standard deviation {:.3e}".format(p_opt[0], p_cov[0,0]))


# impedance RCL

# theoretical curve
R = 512.4
C = 22e-9
L = 42.6e-3
freq = np.linspace(100, 90000, 1000)
angular_freq = 2 * np.pi * freq
Z_RCL = np.sqrt(R ** 2 + (angular_freq * L - 1 / (angular_freq * C)) ** 2)

data = pd.read_csv(r"C:\Users\lenovo\Desktop\courses\PHY224\Experiments\LCR\impedance RCL.csv", delimiter=',',
                   skiprows=1)
data['angular_freq'] = 2 * np.pi * data['freq']
data['Z_RCL'] = data.V1 / data.V2 * R
fig, ax = plt.subplots()
ax.plot(data['angular_freq'], data['Z_RCL'], label="observed")
# ax.plot(angular_freq, Z_RCL, label="theoretical")
ax.legend()
ax.set_xlabel('Angular Frequency')
ax.set_ylabel('Z')
plt.show()

sub_data = data[0:6]
def impedance_rl(angular_freq, L):
    R = 512.4
    Z_RL = np.sqrt(R ** 2 + ( (angular_freq * L)) ** 2)
    return Z_RL

xdata = sub_data['angular_freq']
ydata = sub_data['Z_RL']
reading_error = 0.03  # the error is chosen to be the "chop-off" precision error
sigma_x = np.ones_like(xdata) * 0.000001  # the error is chosen to be the "chop-off" precision error
sigma_y = ydata * np.sqrt((sub_data['V1']*reading_error / sub_data['V1']) ** 2 + (sub_data['V2']*reading_error / sub_data['V2']) ** 2)

p_opt, p_cov = curve_fit(impedance_rl, xdata, ydata, (0.04), sigma_y, True)

fig, ax = plt.subplots()
ax.plot(xdata, impedance_rl(xdata, *p_opt), 'r-',
        label='fit: C=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error",linestyle='None', marker=".")
ax.set_title('RC circuit')
ax.set_xlabel('freqency')
ax.set_ylabel('Z')
ax.legend()
plt.show()
print("The capacitance estimated is {:.3e} with standard deviation {:.3e}".format(p_opt[0], p_cov[0,0]))
