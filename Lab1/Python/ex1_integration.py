""" Exercise 1 - integration """

from typing import Callable, Any
import numpy as np
from scipy.integrate import odeint, ode, RK45

import farms_pylog as pylog
from cmcpack import Result

# pylint: disable=invalid-name


def example_integrate(
        x0: np.array,
        time_max: float,
        time_step: float
) -> Result:
    """ Example to show how to use Python

    Note that the Result class takes x and time as input, and is used to
    facilitate the plotting of the results, where x and time are lists or
    arrays

    Additionally, using:

    r = Result(x, t)

    You can then use r.state or r.time to extract the original x and t

    You can then plot the state in function of time with:
    r.plot_state(figure='Name of figure')
    """
    time = np.arange(0, time_max, time_step)
    x = np.zeros([len(time), len(x0)])
    for i, ti in enumerate(time[:-1]):
        x[i+1] = x[i] + 1e-2*ti
    # for i in range(len(time[:-1])):  # Also possible
    #     x[i+1] = x[i] + 0.2
    return Result(x, time)


def euler_integrate(
        fun: Callable,
        x0: np.array,
        time_max: float,
        time_step: float
) -> Result:
    """ Integrate function using Euler method

    - fun: Function df/dt = f(x, t) to integrate
    - x0: Initial state
    - time_tot: Total time to simulate [s]
    - time_step: Time step [s]

    For loop in Python:

    >>> a = [0, 0, 0]  # Same as [0 for _ in range(3)] (List comprehension)
    >>> for i in range(3):
    ...     a[i] = i
    >>> print(a)
    [0, 1, 2]

    Creating a matrix of zeros in python:

    >>> state = np.zeros([3, 2])
    >>> print(state)
    [[0. 0.]
     [0. 0.]
     [0. 0.]]

    For loop for a state in python

    >>> state = np.zeros([3, 2])
    >>> state[0, 0], state[0, 1] = 1, 2
    >>> for i, time in enumerate([0, 0.1, 0.2]):
    ...     state[i, 0] += time
    ...     state[i, 1] -= time
    >>> print(state)
    [[ 1.   2. ]
     [ 0.1 -0.1]
     [ 0.2 -0.2]]

    Make sure to use the Result class similarly to the example_integrate
    found above (i.e. Result(x, time))
    """
    time = np.arange(0, time_max, time_step)
    x = np.zeros([len(time), len(x0)])
    # COMPLETE CODE
    pylog.warning('Euler integration must be implemented')
    return None


def ode_integrate(
        fun: Callable,
        x0: np.array,
        time_max: float,
        time_step: float
) -> Result:
    """ Integrate function using Lsoda method

    - fun: Function df/dt = f(x, t) to integrate
    - x0: Initial state
    - time_tot: Total time to simulate [s]
    - time_step: Time step [s]

    Use odeint from the scipy library for integration

    Make sure to then use the Result class similarly to the example_integrate
    found above (i.e. Result(x, time))
    """
    time = np.arange(0, time_max, time_step)
    # COMPLETE CODE
    pylog.warning('ODE integration must be implemented')
    return None


def ode_integrate_rk(
        fun_rk: Callable,
        x0: np.array,
        time_max: float,
        time_step: float
) -> Result:
    """
    Integrate function using Runge Kutta 4th order method.
    See euler_integrate above for a reference to the inputs
    and outputs of this function.
    """
    time = np.arange(0, time_max, time_step)
    x = np.zeros([len(time), len(x0)])
    # COMPLETE CODE
    pylog.warning('Runge Kutta integration must be implemented')
    return None


def ode_integrate_dopri(
        fun_rk: Callable,
        x0: np.array,
        time_max: float,
        time_step: float
) -> Result:
    """ Integrate function using dopri method

    - fun: Function df/dt = f(x, t) to integrate
    - x0: Initial state
    - time_tot: Total time to simulate [s]
    - time_step: Time step [s]

    For Runge-Kutta, use:
    solver = ode(fun)
    solver.set_integrator('dopri5')
    solver.set_initial_value(x0, time[0])
    xi = solver.integrate(ti)

    Note that solver.integrate(ti) only integrates for one time step at time ti
    (where ti is the current time), so you will need to use a for loop

    Make sure to use the Result class similarly to the example_integrate
    found above (i.e. Result(x, time))
    """
    time = np.arange(0, time_max, time_step)
    # COMPLETE CODE
    pylog.warning('ODE integration with RK must be implemented')
    return None


def plot_integration_methods(**kwargs):
    """ Plot integration methods results """
    # pylint: disable=too-many-locals
    pylog.info('Plotting integration results')

    # Results
    analytical = kwargs.pop('analytical', None)
    euler = kwargs.pop('euler', None)
    ode_fun = kwargs.pop('ode', None)
    ode_rk = kwargs.pop('ode_rk', None)
    ode_dopri = kwargs.pop('ode_dopri', None)
    euler_ts_small = kwargs.pop('euler_ts_small', None)

    # Figures
    fig_all = kwargs.pop(
        'figure',
        'Integration methods'.replace(' ', '_')
    )
    fig_ts = 'Integration methods smaller ts'.replace(' ', '_')
    fig_euler = 'Euler integration'.replace(' ', '_')
    d = '.'

    # Time steps information
    et_val = kwargs.pop('euler_timestep', None)
    ets_val = kwargs.pop('euler_timestep_small', None)
    et = f' (ts={et_val})' if et_val is not None else ''
    ets = f' (ts={ets_val})' if ets_val is not None else ''

    assert not kwargs, kwargs

    # Analytical
    if analytical is not None:
        analytical.plot_state(figure=fig_euler, label='Analytical', marker=d)

    # Euler
    if euler is not None:
        euler.plot_state(figure=fig_euler, label='Euler'+et, marker=d)

    # ODE
    ls = ' '
    ode_plot = False
    if ode_fun is not None:
        if ode_plot is False:
            analytical.plot_state(figure=fig_all, label='Analytical', marker=d)
            euler.plot_state(figure=fig_all, label='Euler'+et, marker=d)
            ode_plot = True
        ode_fun.plot_state(
            figure=fig_all,
            label='LSODA',
            linestyle=ls,
            marker='x')

    # ODE Runge-Kutta
    if ode_rk is not None:
        if ode_plot is False:
            analytical.plot_state(figure=fig_all, label='Analytical', marker=d)
            euler.plot_state(figure=fig_all, label='Euler'+et, marker=d)
        ode_rk.plot_state(figure=fig_all, label='RK', linestyle=ls, marker=d)

    # ODE dopri
    if ode_dopri is not None:
        if ode_plot is False:
            analytical.plot_state(figure=fig_all, label='Analytical', marker=d)
            euler.plot_state(figure=fig_all, label='Euler'+et, marker=d)
        ode_dopri.plot_state(
            figure=fig_all,
            label='dopri',
            linestyle=ls,
            marker='>')

    # Euler with lower time step
    if euler_ts_small is not None:
        euler_ts_small.plot_state(
            figure=fig_ts,
            label='Euler'+ets
        )
        analytical.plot_state(figure=fig_ts, label='Analytical', marker=d)

