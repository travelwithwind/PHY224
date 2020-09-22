import numpy as np
import matplotlib.pyplot as plt


# data = np.loadtxt("position.txt", skiprows=2, delimiter="\t")


# class SpringMass():
#     """
#     Spring-Mass Oscillation model
#
#     === Attributes ===
#     v: array of velocity
#     y: array of position
#     k: spring constant measured, implied_k = angular_freq**2*mass
#     period: period of oscillation
#     mass: mass of weight
#     angular_freq: also known as omega, calulated as 2*np.pi/period or sqrt(k/mass)
#     t: array of time stamps
#     energy_k: kinetic energy
#     energy_u: potential energy
#     energy_tot: total energy
#     """
#
#     def __init__(self, v0, y0, mass, period):
#         self.v = [v0]
#         self.y = [y0]
#         self.mass = mass
#         self.period = period
#         self.angular_freq = 2 * np.pi / period
#         self.k = self.angular_freq ** 2 * mass
#         self.t = None
#
#         self.energy_k = [0.5 * self.mass * self.v[-1] ** 2]
#         self.energy_u = [0.5 * self.k * self.y[-1] ** 2]
#         self.energy_tot = [self.energy_k[-1] + self.energy_u[-1]]
#         self.gamma = gamma
#
#     def integrate(self, dt, t=10, method="forward"):
#         """
#         update velocity and position based on Forward Euler
#         calculate
#         :param dt: step size for integration
#         :param t: range of time, default is 10 seconds
#         :param method: forward or symplectic
#         """
#         t = np.arange(0, t, dt)
#         for i in t[:-1]:  # the last time point is not used because Forward euler uses the starting values
#             self.v.append(self.v[-1] - dt * (self.angular_freq ** 2) * self.y[-1] )
#             if method == "forward":
#                 self.y.append(self.y[-1] + dt * self.v[-2])
#             if method == "symplectic":
#                 self.y.append(self.y[-1] + dt * self.v[-1])
#             self.energy_k.append(0.5 * self.mass * self.v[-1] ** 2)
#             self.energy_u.append(0.5 * self.k * self.y[-1] ** 2)
#             self.energy_tot.append(self.energy_k[-1] + self.energy_u[-1])
#
#         self.t = t
#
#     def plot(self, y, x):
#         """
#         plot according to specify x and y, for example
#         position vs. time
#         velocity vs. time
#         velocity vs. position
#         :param x: can be any of position, time, velocity, energy, potential energy,kinetic energy
#         :param y: can be any of position, time, velocity, energy, potential energy,kinetic energy
#         """
#         mapping = {"position": self.y, "velocity": self.v, "time": self.t,
#                    "energy": self.energy_tot, "kinetic energy": self.energy_k,
#                    "potential energy": self.energy_u}
#
#         fig, ax = plt.subplots()
#         ax.plot(mapping[x], mapping[y], "red")
#         ax.set_xlabel(x)
#         ax.set_ylabel(y)
#         ax.set_title(y + " vs " + x)
#         plt.show()



####### undampped ##########
# stepsize = 0.01  # [second]
# T = 10  # [second]
# mass = 0.2  # [kg]
# start_velocity = 0
# start_position = -0.1  # 10cm below equilibrium position
# period = 0.733  # [second]
# gamma = 0.035
#
# model = SpringMass(start_velocity, start_position, mass, period, gamma)
# model.integrate(stepsize, method="symplectic")
# model.plot("position", "time")
# model.plot("velocity", "time")
# model.plot("velocity", "position")
#
# model.plot("potential energy", "time")
# model.plot("kinetic energy", "time")
# model.plot("energy", "time")



class SpringMass_damped():
    """
    Spring-Mass Oscillation model

    === Attributes ===
    v: array of velocity
    y: array of position
    k: spring constant measured, implied_k = angular_freq**2*mass
    period: period of oscillation
    mass: mass of weight
    angular_freq: also known as omega, calulated as 2*np.pi/period or sqrt(k/mass)
    t: array of time stamps
    energy_k: kinetic energy
    energy_u: potential energy
    energy_tot: total energy
    gamma: damping coefficient
    """

    def __init__(self, v0, y0, mass, k, gamma=0):
        self.v = [v0]
        self.y = [y0]
        self.mass = mass
        self.period = period
        self.omega = np.sqrt(k/mass)
        self.k = k
        self.t = None

        self.energy_k = [0.5 * self.mass * self.v[-1] ** 2]
        self.energy_u = [0.5 * self.k * self.y[-1] ** 2]
        self.energy_tot = [self.energy_k[-1] + self.energy_u[-1]]
        self.gamma = gamma

    def integrate(self, dt, t=10, method="forward"):
        """
        update velocity and position based on Forward Euler
        calculate
        :param dt: step size for integration
        :param t: range of time, default is 10 seconds
        :param method: forward or symplectic
        """
        t = np.arange(0, t, dt)
        for i in t[:-1]:  # the last time point is not used because Forward euler uses the starting values
            self.v.append(self.v[-1] - dt * (self.omega ** 2) * self.y[-1] - self.gamma*self.v[-1])
            if method == "forward":
                self.y.append(self.y[-1] + dt * self.v[-2])
            if method == "symplectic":
                self.y.append(self.y[-1] + dt * self.v[-1])
            self.energy_k.append(0.5 * self.mass * self.v[-1] ** 2)
            self.energy_u.append(0.5 * self.k * self.y[-1] ** 2)
            self.energy_tot.append(self.energy_k[-1] + self.energy_u[-1])

        self.t = t

    def plot(self, y, x):
        """
        plot according to specify x and y, for example
        position vs. time
        velocity vs. time
        velocity vs. position
        :param x: can be any of position, time, velocity, energy, potential energy,kinetic energy
        :param y: can be any of position, time, velocity, energy, potential energy,kinetic energy
        """
        mapping = {"position": self.y, "velocity": self.v, "time": self.t,
                   "energy": self.energy_tot, "kinetic energy": self.energy_k,
                   "potential energy": self.energy_u}

        fig, ax = plt.subplots()
        ax.plot(mapping[x], mapping[y], "red")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(y + " vs " + x)
        plt.show()



####### damped ##########
stepsize = 0.01  # [second]
T = 10  # [second]
mass = 0.2167  # [kg]
start_velocity = 0
start_position = -0.1  # 10cm below equilibrium position
k = 14.7
period = 0.7623  # [second], from question 10
gamma = 0.01   # from question 11

model = SpringMass_damped(start_velocity, start_position, mass, k, gamma)
model.integrate(stepsize, method="symplectic")
model.plot("position", "time")
model.plot("velocity", "time")
model.plot("velocity", "position")

model.plot("potential energy", "time")
model.plot("kinetic energy", "time")
model.plot("energy", "time")




#
#
#
# fig, a1 = plt.subplots()
#
# a1.plot(t, v, "red", label="velocity")
# a1.set_ylabel('velocity(m/s)')
# a1.set_title("title")
#
# a2 = a1.twinx()
# a2.plot(t, y, label="position")
# a2.set_ylabel('position(cm)')
#
# fig.legend()
