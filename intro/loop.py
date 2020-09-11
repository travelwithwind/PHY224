# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

#The raw data for lengths are in cm, that for time are in seconds

################loop1 ###########
h1_raw = np.array([[4.16, 4.1, 3.78],
                   [4.06, 3.9, 4.24],
                   [4.0, 4.15, 3.95],
                   [4.03, 4.22, 4.22]])

h1_period = h1_raw/3

h1_mean = np.mean(h1_period, axis=1)
h1_std = np.std(h1_period, axis=1)

e1 = 1.96 * h1_std/np.sqrt(h1_raw.shape[1])

x = np.array([1, 2, 3, 4])
fig, ax = plt.subplots()
ax.errorbar(x, h1_mean, e1, linestyle='None', marker='^')
ax.set_title('hoop1 period')
ax.set_xticks(x)
ax.set_xticklabels(["pivot 1", "pivot 2", "pivot 3", "pivot 4"])
plt.show()

################loop2 ###########
h2_raw = np.array([[2.82, 2.69, 2.87],
                   [2.69, 2.75, 2.60],
                   [2.62, 2.85,2.87],
                   [2.75, 2.69, 2.81]])

h2_period = h2_raw/3

h2_mean = np.mean(h2_period, axis=1)
h2_std = np.std(h2_period, axis=1)
e2 = 1.96 * h1_std/np.sqrt(h1_raw.shape[1])

fig, ax = plt.subplots()
plt.errorbar(x, h2_mean, e2, linestyle='None', marker='^')
ax.set_title('hoop2 period')
ax.set_xticks(x)
ax.set_xticklabels(["pivot 1", "pivot 2", "pivot 3", "pivot 4"])
plt.show()

################loop3 ###########
h3_raw = np.array([[1.88, 1.97, 2.12],
                   [1.88, 1.97, 1.93],
                   [2.03, 2.03, 1.90],
                   [2.2, 1.96, 2]])

h3_period = h3_raw/3

h3_mean = np.mean(h3_period, axis=1)
h3_std = np.std(h3_period, axis=1)
e3 = 1.96 * h3_std/np.sqrt(h3_raw.shape[1])

fig, ax = plt.subplots()
ax.errorbar(x, h3_mean, e3, linestyle='None', marker='^')
ax.set_title('hoop3 period')
ax.set_xticks(x)
ax.set_xticklabels(["pivot 1", "pivot 2", "pivot 3", "pivot 4"])
plt.show()


################loop4 ###########
h4_raw = np.array([[1.37, 1.41, 1.56],
                   [1.59, 1.38, 1.4],
                   [1.56, 1.44, 1.28],
                   [1.43, 1.37, 1.4]])

h4_period = h4_raw/3

h4_mean = np.mean(h4_period, axis=1)
h4_std = np.std(h4_period, axis=1)
e4 = 1.96 * h4_std/np.sqrt(h4_raw.shape[1])

fig, ax = plt.subplots()
ax.errorbar(x, h4_mean, e4, linestyle='None', marker='^')
ax.set_title('hoop4 period')
ax.set_xticks(x)
ax.set_xticklabels(["pivot 1", "pivot 2", "pivot 3", "pivot 4"])
plt.show()

####### global mean#########
global_mean = np.empty(4)
global_std_error = np.empty(4)
aggregate_data = [h1_period, h2_period, h3_period, h4_period]
for i in range(4):
    global_mean[i] =np.mean(aggregate_data[i] )
    global_std_error[i] = np.std(aggregate_data[i])

diameter = np.array([45.1, 21, 10.7, 5.5])/100
g = diameter/(global_mean/(2*np.pi))**2
g_std = (g*2*global_std_error)**2

fig, ax = plt.subplots()
ax.plot(x, g)
ax.errorbar(x, g, g_std, linestyle='None', marker='^')
ax.set_xticks(x)
ax.set_title('Gravitational Acceleration g')
ax.set_xticklabels(["loop 1", "loop 2", "loop 3", "loop 4"])
plt.show()

final_mean = np.mean(g)
final_std_error  = np.sum(g_std)/np.sqrt(len(g_std))
print("g={:.2f}, with standard error {:.2f}".format(final_mean,final_std_error))