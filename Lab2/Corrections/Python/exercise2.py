""" Lab 4 """

import matplotlib.pyplot as plt
import numpy as np

import farms_pylog as pylog
from cmcpack import DEFAULT, integrate, parse_args
from pendulum_system import PendulumSystem
from system_animation import SystemAnimation
from system_parameters import PendulumParameters

DEFAULT['label'] = [r'$\theta$ [rad]', r'$d\theta/dt$ [rad/s]']

# pylint: disable=invalid-name


def pendulum_integration(state: np.array, time: float, *args):
    """ Function for system integration """
    pendulum = args[0]
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=0.0,
    )[:, 0]


def pendulum_perturbation(
        state: np.array,
        time: float,
        *args,
) -> np.array:
    """ Function for system integration with perturbations.
    Setup your time based system pertubations within this function.
    The pertubation can either be applied to the states or as an
    external torque.
    """
    pendulum = args[0]
    pert_type = 'speed' if len(args) == 1 else args[1]
    torque = 0.0
    if 5 < time < 5.1:
        pylog.info('Applying state perturbation to pendulum_system')
        if pert_type == 'speed':
            state[1] = 2.
        elif pert_type == 'angle':
            state[0] = -1.
        elif pert_type == 'torque':
            torque = 4.
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=torque,
    )[:, 0]



def pendulum_limit_cycle(x0: np.array, time: float, *args):
    """Pendulum limit cycle"""
    # Initialize the parameters without damping
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0

    pendulum.parameters.b2 = 0.0
    pylog.info(pendulum.parameters.show_parameters())
    pylog.info(
        '1a. Running pendulum_system with'
        ' springs to study limit cycle behavior'
    )
    title = '{} Limit Cycle(x0 = {})'
    res = integrate(pendulum_perturbation, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def pendulum_spring_constant(x0: np.array, time: float, *args):
    """Pendulum with spring constant"""
    # Initialize the parameters to bias spring 1
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 0.1
    pendulum.parameters.k2 = 0.1
    pylog.info(
        '1b. Running pendulum_system for analysing role of spring constant'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Spring Constant 1(x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))

    # Initialize the parameters to bias spring 2
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 100.
    pendulum.parameters.k2 = 100.
    pylog.info(
        '1b. Running pendulum_system for analysing role of spring constant'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Spring Constant 2(x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))

    # Initialize the pendulum.parameters to bias spring 1
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 1.0
    pendulum.parameters.k2 = 100.0
    pylog.info(
        '1b. Running pendulum_system for'
        ' analysing role of variable spring constant'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Variable Spring Constant 1(x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def pendulum_spring_reference(x0: np.array, time: float, *args):
    """Pendulum spring reference"""
    # Initialize the parameters to bias spring 1 reference angle
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 10.0
    pendulum.parameters.k2 = 10.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-10.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(10.0)

    pylog.info(
        '1b. Running pendulum_system for analysing role of spring reference'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Spring Reference 1(x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))

    # Initialize the pendulum.parameters to bias spring 2 reference angle
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 10.0
    pendulum.parameters.k2 = 10.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-75.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(75.)
    pylog.info(
        '1b. Running pendulum_system for analysing role of spring reference')
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Spring Reference 2(x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))

    # Initialize the pendulum.parameters to bias spring 2 reference angle
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 10.0
    pendulum.parameters.k2 = 10.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(0.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(75.0)

    pylog.info(
        '1c. Running pendulum_system for'
        ' analysing role of variable spring reference'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Variable Spring Reference 1(x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def pendulum_spring_damper(x0: np.array, time: float, *args):
    """ Function to analyse the pendulum spring damper system"""
    pendulum = args[0]
    pendulum.parameters.b1 = 0.5
    pendulum.parameters.b2 = 0.5
    pendulum.parameters.k1 = 50.0
    pendulum.parameters.k2 = 50.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-45.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(45.0)
    pylog.info(
        '20. Running pendulum_system for'
        ' analysing role of spring and damper muscle'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Spring Damper (x0 = {})'
    res = integrate(pendulum_perturbation, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def pendulum_spring_large_damper(x0: np.array, time: float, *args):
    """
    Function to analyse the pendulum spring damper system with large damping
    """
    pendulum = args[0]
    pendulum.parameters.b1 = 5.
    pendulum.parameters.b2 = 5.
    pendulum.parameters.k1 = 5.0
    pendulum.parameters.k2 = 5.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(-45.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(45.0)
    pylog.info(
        '20. Running pendulum_system for analysing the role of large damping'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Spring and Large Damping (x0 = {})'
    res = integrate(pendulum_perturbation, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def pendulum_set_position(x0: np.array, time: float, *args):
    """ Function to analyse the pendulum spring damper system"""
    pendulum = args[0]
    pendulum.parameters.b1 = 1.
    pendulum.parameters.b2 = 1.
    pendulum.parameters.k1 = 50.0
    pendulum.parameters.k2 = 50.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(0.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(65.6)

    pylog.info(
        '1b. Running pendulum_system to set fixed position')
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Pendulum Fixed Position (x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    pylog.debug('Position : %s', np.rad2deg(res.state[-1]))
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def pendulum_set_position_large_damper(
        x0: np.array,
        time: float,
        *args,
):
    """
    Function to analyse the pendulum spring damper system with large damping
    """
    pendulum = args[0]
    pendulum.parameters.b1 = 10.
    pendulum.parameters.b2 = 10.
    pendulum.parameters.k1 = 50.0
    pendulum.parameters.k2 = 50.0
    pendulum.parameters.s_theta_ref1 = np.deg2rad(0.0)
    pendulum.parameters.s_theta_ref2 = np.deg2rad(65.6)

    pylog.info(
        '1b. Running pendulum_system to set'
        ' fixed position with a large damping term'
    )
    pylog.info(pendulum.parameters.show_parameters())
    title = '{} Pendulum Fixed Position with large damping (x0 = {})'
    res = integrate(pendulum_integration, x0, time, args=args)
    pylog.debug('Position : %s', np.rad2deg(res.state[-1]))
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))



def exercise2():
    """ Exercise 2  """
    pylog.info('Executing Lab 2 : Exercise 1')
    pendulum = PendulumSystem()
    pendulum.parameters = PendulumParameters()

    pylog.info(
        'Find more information about'
        ' Pendulum Parameters in SystemParameters.py'
    )

    # Initialize a new pendulum system with default parameters
    pendulum = PendulumSystem()
    pendulum.parameters = PendulumParameters()

    pylog.info(
        'Find more information about'
        ' Pendulum Parameters in SystemParameters.py'
    )

    # To change a parameter you could so by,
    # >>> pendulum.parameters.L = 1.5
    # The above line changes the length of the pendulum

    # You can instantiate multiple pendulum models with different parameters
    # For example,

    # >>> pendulum_1 = PendulumSystem()
    # >>> pendulum_1.parameters = PendulumParameters()
    # >>> pendulum_1.parameters.L = 0.5
    # >>> parameters_2 = PendulumParameters()
    # >>> parameters_2.L = 0.5
    # >>> pendulum_2 = PendulumSystem(paramters=parameters_2)

    # Above examples shows how to create two istances of the pendulum
    # and changing the different parameters using two different approaches

    pendulum.parameters.k1 = 50
    pendulum.parameters.k2 = 50
    pendulum.parameters.s_theta_ref1 = 1.0
    pendulum.parameters.s_theta_ref2 = 1.0
    pendulum.parameters.b1 = 0.5
    pendulum.parameters.b2 = 0.5

    pylog.warning('Loading default pendulum pendulum.parameters')
    pylog.info(pendulum.parameters.show_parameters())

    # Simulation Parameters
    t_start = 0.0
    t_stop = 10.0
    dt = 0.01
    pylog.warning('Using large time step dt=%s', dt)
    time = np.arange(t_start, t_stop, dt)
    x0 = [0.5, 0.0]

    x0 = [0.5, 0.1]
    pendulum_limit_cycle(x0, time, pendulum)
    pendulum_spring_constant(x0, time, pendulum)
    pendulum_spring_reference(x0, time, pendulum)
    pendulum_spring_damper(x0, time, pendulum)
    pendulum_spring_large_damper(x0, time, pendulum, 'angle')
    pendulum_set_position(x0, time, pendulum)
    pendulum_set_position_large_damper(x0, time, pendulum)

    if DEFAULT['save_figures'] is False:
        plt.show()


if __name__ == '__main__':
    parse_args()
    exercise2()