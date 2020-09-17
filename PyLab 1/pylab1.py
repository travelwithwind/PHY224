import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("position.txt", skiprows=2, delimiter="\t")

dt = 0.01
# initial values
v0 = 0  # velocity
y0 = -10  # poistion
k0 = 0  # kinetic energy
u0 = 50  # spring potential energy

period = 0.733
angular_freq = 2 * np.pi / period

mass = 0.2
implied_k = angular_freq ** 2 * mass


class SpringMass():
    '''
    Spring-Mass Oscillation model

    === Attributes ===
    v: array of velocity
    y: array of position
    k: spring constant measured, implied_k = angular_freq**2*mass
    period: period of oscillation
    mass: mass of weight
    angular_freq: also known as omega, calulated as 2*np.pi/period or sqrt(k/mass)
    t: array of time stamps
    '''

    def __init__(self, v0, y0, mass, period):
        self.v = [v0]
        self.y = [y0]
        self.period = period
        self.angular_freq = 2 * np.pi / period
        self.k = self.angular_freq ** 2 * mass
        self.t = None
        # k=[k0]
        # u=[u0]

    def integrate(self, dt, t=10):
        '''
        update velocity and position based on Forward Euler

        :param dt: step size for integration
        :param t: range of time, default is 10 seconds
        '''
        t = np.arange(0, t, dt)
        for i in t[:-1]:  # the last time point is not used because Forward euler uses the starting values
            self.v.append(self.v[-1] - dt * (self.angular_freq ** 2) * self.y[-1])
            self.y.append(self.y[-1] + dt * self.v[-1])
        self.t = t

    def plot(self, y, x):
        '''
        plot according to specify x and y, for example
        position vs. time
        velocity vs. time
        velocity vs. position
        :param x:
        :param y:
        :return:
        '''
        mapping = {"position": self.y, "velocity": self.v, "time": self.t}

        fig, ax = plt.subplots()
        ax.plot(mapping[x], mapping[y], "red")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(y+" vs " + x)
        plt.show()




model = SpringMass(0,-10, 0.2, 0.733)
model.integrate(0.01)
model.plot("position", "time")
model.plot("velocity", "time")
model.plot("velocity", "position")

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
