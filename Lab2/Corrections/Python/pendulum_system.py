""" Pendulum """

import numpy as np
from system_parameters import PendulumParameters

# pylint: disable=invalid-name,unused-argument,unused-variable


class PendulumSystem:
    """Pendulum model main class.
    The Pendulum system class consists of all the methods to setup
    and simulate the pendulum dynamics. You need to implement the
    relevant pendulum equations in the following functions.

    #: To create a pendulum object with default pendulum parameters
    >>> pendulum = PendulumSystem()
    #: To create a pendulum object with pre defined parameters
    >>> from system_parameters import PendulumParameters
    >>> parameters = PendulumParameters()
    >>> parameters.L = 0.3 #: Refer PendulumParameters for more info
    >>> pendulum = PendulumSystem(parameters=parameters)
    #: Method to get the first order derivatives of the pendulum
    >>> pendulum = PendulumSystem()
    >>> theta = 0.0
    >>> dtheta = 0.0
    >>> time = 0.0
    >>> derivatives = pendulum.pendulum_system(theta, dtheta, time, torque=0.0)
    """

    def __init__(self, parameters: PendulumParameters = PendulumParameters()):
        """ Initialization """
        super().__init__()
        self.origin: np.array = np.array([0.0, 0.0])
        self.theta: float = 0.0
        self.dtheta: float = 0.0
        self.parameters: PendulumParameters = parameters

    def pendulum_equation(
            self,
            theta: float,
            dtheta: float,
            time: float,
            torque: float,
    ) -> np.array:
        """ Pendulum equation d2theta = -g/L*sin(theta)

        Parameters
        ----------
        self: type
            description
        theta: <float>
            Angle [rad]
        dtheta: <float>
            Angular velocity [rad/s]
        time: <float>
            Time [s]
        torque: <float> <Optional>
            External torque
        With the parameters attribute of the class you also have access to
            - g: Gravity constant [m/s**2]
            - m: Mass [kg]
            - L: Length [m]
            - I: Inertia [kg-m**2]
            - sin: np.sin
            - k1 : Spring constant of spring 1 [N m/rad]
            - k2 : Spring constant of spring 2 [N m/rad]
            - s_theta_ref1 : Spring 1 reference angle [rad]
            - s_theta_ref2 : Spring 2 reference angle [rad]
            - b1 : Damping constant damper 1 [N-s/rad]
            - b2 : Damping constant damper 2 [N-s/rad]

        """
        # pylint: disable=too-many-locals
        g, L, m, I, sin, k1, k2, s_theta_ref1, s_theta_ref2, b1, b2, dry = (
            self.parameters.g,
            self.parameters.L,
            self.parameters.m,
            self.parameters.I,
            self.parameters.sin,
            self.parameters.k1,
            self.parameters.k2,
            self.parameters.s_theta_ref1,
            self.parameters.s_theta_ref2,
            self.parameters.b1,
            self.parameters.b2,
            self.parameters.dry
        )


        pendulum_equation = -g * (1. / L) * sin(theta) + torque/I

        spring_equation = min(k1 * (s_theta_ref1 - theta)/I, 0) + \
            max(k2 * (s_theta_ref2 - theta)/I, 0)
        if not dry:
            damper_equation = b1 * dtheta/I + b2 * dtheta/I
        else:
            damper_equation = b1 * dtheta/I + b2 * self.sign(dtheta)/I

        return pendulum_equation + spring_equation - damper_equation



    def sign(self, dtheta: float) -> float:
        """ return sign """
        # return np.sign(dtheta)  # May run into integration instability
        tol = 1e6
        return np.sign(dtheta) if tol*np.abs(dtheta) > 1 else tol*dtheta

    def pendulum_system(
            self,
            theta: float,
            dtheta: float,
            time: float,
            torque: float = 0.0,
    ) ->np.array:
        """ Pendulum """
        return np.array([
            [dtheta],
            [self.pendulum_equation(
                theta, dtheta, time, torque)]  # d2theta
        ])

    def pose(self) -> np.array:
        """Compute the full pose of the pendulum.

        Returns:
        --------
        pose: np.array
            [origin, center-of-mass]

        """
        return np.array(
            [self.origin,
             self.origin + self.link_pose()])

    def link_pose(self) -> np.array:
        """ Position of the pendulum center of mass.

        Returns:
        --------
        link_pose: np.array
            Returns the current pose of pendulum COM

        """

        return self.parameters.L * np.array([
            np.sin(self.theta),
            -np.cos(self.theta),
        ])

    @property
    def state(self) -> np.array:
        """ Get the pendulum state  """
        return [self.theta, self.dtheta]

    @state.setter
    def state(self, value: np.array):
        """"Set the state of the pendulum.

        Parameters:
        -----------
        value: np.array
            Position and Velocity of the pendulum

        """
        self.theta = value[0]
        self.dtheta = value[1]