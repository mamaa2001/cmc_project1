""" Lab 4 """

import matplotlib.pyplot as plt
import numpy as np

import farms_pylog as pylog
from cmcpack import DEFAULT, integrate, parse_args
from cmcpack.plot import bioplot, save_figure
from pendulum_system import PendulumSystem
from system_animation import SystemAnimation
from system_parameters import PendulumParameters

DEFAULT['label'] = [r'$\theta$ [rad]', r'$d\theta/dt$ [rad/s]']

# pylint: disable=invalid-name


def pendulum_integration(state: np.array, time: float, *args):
    """ Function for system integration with no perturbations.
    """
    pendulum = args[0]
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=0.0,
    )[:, 0]



def pendulum_unperturbed(
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
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=0,
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
        # pylog.info('Applying state perturbation to pendulum_system')
        if pert_type == 'speed':
            state[1] = 2.
        elif pert_type == 'angle':
            state[0] = -1.
        elif pert_type == 'torque':
            torque = 4.
    return pendulum.pendulum_system(
        state[0], state[1], time, torque=torque,
    )[:, 0]


def evolution_cases(time: float, *args):
    """ Normal simulation """
    # Initialize the parameters without damping
    pylog.info('Evolution with basic paramater')
    x0_cases = [
        ['Normal', [0.1, 0]],
        ['Stable', [0.0, 0.0]],
        ['Unstable', [np.pi, 0.0]],
        ['Multiple loops', [0.1, 10.0]]
    ]
    pendulum = args[0]
    pendulum.parameters.b1 = 0.15
    pendulum.parameters.b2 = 0.15
    pendulum.parameters.k1 = 0
    pendulum.parameters.k2 = 0

    title = '{} case {} (x0={})'
    for name, x_0 in x0_cases:
        res = integrate(pendulum_unperturbed, x_0, time, args=args)
        res.plot_state(title.format(name, 'state', x_0))
        res.plot_phase(title.format(name, 'phase', x_0))



def fixed_point_types(time: float, *args):
    """Pendulum limit cycle"""
    # Initialize the parameters without damping
    pendulum = args[0]
    pendulum.parameters.k1 = 0
    pendulum.parameters.k2 = 0

    pylog.info('Evolution with modified parameters')

    figname = 'Fixed point types'
    labs = ['Underdamped', 'Critically damped', 'Overdamped']
    param_cases = [0.5, 1.0, 1.5]
    x0 = [np.pi/2, 0.0]
    states = []
    for pfact in param_cases:
        pendulum.parameters.b1 = pfact * np.sqrt(pendulum.parameters.g/pendulum.parameters.L)
        pendulum.parameters.b2 = pfact * np.sqrt(pendulum.parameters.g/pendulum.parameters.L)
        res = integrate(pendulum_unperturbed, x0, time, args=args)
        states.append(res.state)

    states = np.array(states)

    bioplot(
        data_x=states[:, :, 0].T,
        data_y=time,
        figure=figname + '_Temporal evolution',
        label=labs,
    )

    plt.figure(figname + '_Phase plot')
    for i, lab in enumerate(labs):
        plt.plot(states[i, :, 0], states[i, :, 1], linewidth=2.0, label=lab)
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$d\theta/dt$ [rad/s]')
    plt.legend()
    plt.grid('True')
    save_figure(figname + '_Phase plot')


def evolution_no_damping(x0: np.array, time: float, *args):
    """ No damping simulation """
    pylog.info('Evolution with no damping')
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 0
    pendulum.parameters.k2 = 0

    title = '{} without damping (x0={})'
    res = integrate(pendulum_unperturbed, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def evolution_perturbation(x0: np.array, time: float, *args):
    """ Perturbation and no damping simulation """
    pylog.info('Evolution with perturbations')
    pendulum = args[0]
    pendulum.parameters.b1 = 0.0
    pendulum.parameters.b2 = 0.0
    pendulum.parameters.k1 = 0
    pendulum.parameters.k2 = 0

    title = '{} with perturbation (x0={})'
    res = integrate(pendulum_perturbation, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))


def evolution_dry(x0: np.array, time: float, *args):
    """ Dry friction simulation """
    pylog.info('Evolution with dry friction')
    pendulum = args[0]
    pendulum.parameters.b1 = 0.05
    pendulum.parameters.b2 = 0.05
    pendulum.parameters.k1 = 0
    pendulum.parameters.k2 = 0
    pendulum.parameters.dry = True

    title = '{} with dry friction (x0={})'
    res = integrate(pendulum_unperturbed, x0, time, args=args)
    res.plot_state(title.format('State', x0))
    res.plot_phase(title.format('Phase', x0))





def exercise1():
    """ Exercise 1  """
    pylog.info('Executing Lab 4 : Exercise 1')
    pendulum = PendulumSystem()
    pendulum.parameters = PendulumParameters()

    pylog.info(
        'Find more information about'
        ' Pendulum Parameters in SystemParameters.py'
    )
    pylog.info('Executing Lab 4 : Exercise 1')

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

    res = integrate(pendulum_integration, x0, time, args=(pendulum,))
    pylog.info('Instructions for applying pertubations')
    # Use pendulum_perturbation method to apply pertubations
    # Define the pertubations inside the function pendulum_perturbation
    # res = integrate(pendulum_perturbation, x0, time, args=(pendulum,))
    res.plot_state('State')
    res.plot_phase('Phase')

    fixed_point_types(time, pendulum)
    evolution_cases(np.arange(0, 30, dt), pendulum)
    evolution_no_damping([0.1, 0], np.arange(0, 30, dt), pendulum)
    evolution_perturbation([0.1, 0], np.arange(0, 30, dt), pendulum, 'torque')
    evolution_dry([0.1, 0], np.arange(0, 30, dt), pendulum)



    if DEFAULT['save_figures'] is False:
        plt.show()


if __name__ == '__main__':
    parse_args()
    exercise1()