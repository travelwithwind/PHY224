import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


h0 = 0.149   # the initial reading of ruler without using any weight and current
d0 = 0.007 # the initial distance between wires without using any weight and current
L = 1.9 # distance between scope and mirror
l = 0.29   # length of wire
m = 0.215  # distance between wire and mirror


def read_data(filename):
    data = pd.read_csv(filename, delimiter=',', skiprows=2)
    data.columns=["transformer","lower", "upper", "ruler"]
    data["current"] = (data["lower"]+ data["upper"])/2
    data['error'] = data["current"]- data["lower"]
    data["ruler"] = data["ruler"]/100 #divide by 100 to convert cm to m
    data["dH"] = data["ruler"] - h0
    data["dTheta"] = np.arctan(data["dH"]/L)
    data["dAlpha"] = data["dTheta"] /2
    data['d'] = np.tan(data["dAlpha"])*m + d0
    return data


weight0=read_data("weight0.csv")
weight20=read_data("weight20mg.csv")
weight40=read_data("weight40mg.csv")
weight60=read_data("weight60mg.csv")

fig, ax = plt.subplots()
ax.plot(weight0["current"], weight0['d'],  label="weight0")
ax.plot(weight20["current"], weight20['d'], label="weight20")
ax.plot(weight40["current"], weight40['d'], label="weight40")
ax.plot(weight60["current"], weight60['d'], label="weight60")
ax.set_title('distance against current')
ax.set_xlabel('current(A)')
ax.set_ylabel('distance(m)')

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0.004, 0.01, 0.001)
minor_ticks = np.arange(0.004, 0.01, 0.0005)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(3, 11, 1)
minor_ticks = np.arange(3, 11, 0.1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)

ax.grid(b=True, which="both")
ax.legend()
plt.show()


def weight(current, f0, mu):
    d = 0.0075
    F = mu * (current ** 2) * l / (2*np.pi*d)
    return F-f0


# estimate the uncertainties in d
sigma_dH = 0.00005
sigma_L = 0.02

df_temp = weight0[["current", "dH","d"]].iloc[0:2]
sigma11 = df_temp.d[0]*np.sqrt((sigma_dH/df_temp.dH[0])**2+(sigma_L/L)**2)
sigma12 = df_temp.d[1]*np.sqrt((sigma_dH/df_temp.dH[1])**2+(sigma_L/L)**2)
sigma1 = np.mean((sigma11, sigma12))

df_temp2 = weight20[["current", "dH","d"]].iloc[3:5]
sigma21 = df_temp2.d[3]*np.sqrt((sigma_dH/df_temp2.dH[3])**2+(sigma_L/L)**2)
sigma22 = df_temp2.d[4]*np.sqrt((sigma_dH/df_temp2.dH[4])**2+(sigma_L/L)**2)
sigma2 = np.mean((sigma21, sigma22))

df_temp3 = weight40[["current", "dH","d"]].iloc[[8,9]]
sigma31 = df_temp3.d[8]*np.sqrt((sigma_dH/df_temp3.dH[8])**2+(sigma_L/L)**2)
sigma32 = df_temp3.d[9]*np.sqrt((sigma_dH/df_temp3.dH[9])**2+(sigma_L/L)**2)
sigma3 = np.mean((sigma31, sigma32))

df_temp4 = weight60[["current", "dH","d"]].iloc[-1]
sigma4 = df_temp4.d*np.sqrt((sigma_dH/df_temp4.dH)**2+(sigma_L/L)**2)

sigma = [sigma1, sigma2, sigma3, sigma4]



# curve fitting to find mu
dat2 = pd.DataFrame(data=dict(weight=[0,20e-6,40e-6,60e-6], current=[4.3, 6, 9.2, 10.3])) # four points read off from plot
dat2["cur_sqr"] = dat2['current']**2

p_opt, p_cov = curve_fit(weight,  dat2['current'], dat2["weight"], sigma=sigma)

fig, ax = plt.subplots()
ax.plot(dat2['current'], weight(dat2['current'], *p_opt), 'r-',
        label='fit: f0=%.3e, mu=%5.3e' % tuple(p_opt))
ax.errorbar(dat2['current'], dat2['weight'], yerr=sigma , linestyle='None', label="error", marker=".")
plt.legend()
plt.title("weight vs current")
ax.set_ylabel("weight(kg)")
ax.set_xlabel("current(A)")


# chi2 test for goodness of fit
def chi2(xdata, ydata, measurement_error, model_function):
    p_opt, p_cov = curve_fit(model_function, xdata, ydata, sigma=measurement_error)
    n = len(p_opt)
    N = len(xdata)
    v = N - n
    expected = model_function(xdata, *p_opt)
    statistics = sum(((ydata - expected) / measurement_error) ** 2) / v
    return statistics

chi2(dat2['current'], dat2["weight"], sigma, weight)
