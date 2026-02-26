""" Lab 1 - Exercise 1 """

import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from cmcpack import parse_args, Result

from ex1_functions import function, function_rk, analytic_function
from ex1_integration import (
    example_integrate,
    euler_integrate,
    ode_integrate,
    ode_integrate_rk,
    ode_integrate_dopri,
    plot_integration_methods,
)
from ex1_errors import compute_error

# pylint: disable=invalid-name


def exercise1(clargs):
    """ Exercise 1 """
    # pylint: disable=too-many-locals

    # Setup
    pylog.info('Running exercise 1')

    # Setup
    time_max = 5  # Maximum simulation time
    time_step = 0.2  # Time step for ODE integration in simulation
    x0 = np.array([1.])  # Initial state

    # Integration methods (Exercises 1.a - 1.d)
    pylog.info('Running function integration using different methods')

    # Example
    pylog.debug('Running example plot for integration (remove)')
    example = example_integrate(x0, time_max, time_step)
    example.plot_state(figure='Example', label='Example', marker='.')

    # Analytical (1.a)
    time = np.arange(0, time_max, time_step)  # Time vector
    x_a = analytic_function(time)
    analytical = Result(x_a, time) if x_a is not None else None

    # Euler (1.b)
    euler = euler_integrate(function, x0, time_max, time_step)

    # ODE (1.c)
    ode = ode_integrate(function, x0, time_max, time_step)

    # ODE Runge-Kutta (1.c)
    ode_rk = ode_integrate_rk(function_rk, x0, time_max, time_step)

    # ODE Dopri (1.c)
    ode_dopri = ode_integrate_dopri(function_rk, x0, time_max, time_step)

    # Euler with lower time step (1.d)
    pylog.warning('Euler with smaller ts must be implemented')
    euler_time_step = None
    euler_ts_small = (
        euler_integrate(function, x0, time_max, euler_time_step)
        if euler_time_step is not None
        else None
    )

    # Plot integration results
    plot_integration_methods(
        analytical=analytical,
        euler=euler,
        ode=ode,
        ode_rk=ode_rk,
        ode_dopri=ode_dopri,
        euler_ts_small=euler_ts_small,
        euler_timestep=time_step,
        euler_timestep_small=euler_time_step,
    )
    # Error analysis (Exercise 1.e)
    pylog.warning('Error analysis must be implemented')

    # To compute and plot integration error authors can use compute_error()
    # check ex1_errors.py for implementation details.
    # The following input params can be provided:
    #     - func:
    #         reference of the function that needs to be integrated
    #         e.g. function / function_rk
    #     - analytic_function:
    #         reference for analytical function
    #         i.e. analytic_function
    #     - integration_function:
    #         reference to the integrator
    #         eg. euler_integrate / ode_integrate / ode_integrate_rk
    #         check ex1_integration.py
    #     - x0:
    #         initial value for function to be integrated
    #     - dt_list:
    #         list of different timesteps for which error should be calculated
    #         [0.001, 0.005, .01, 0.05]
    #         alternatively can use numpy logspace function
    #     - time_max:
    #         maximum integration time, default val:5
    #     - label:
    #         label of the plot
    #     - figure:
    #         name of the figure
    #     - n:
    #         kind of error L1, L2 norm, range of input=[0, 1, 2 ..]
    #         `0` gives the max of the error `1` gives L1 norm, check
    #         error() in ex1_errors.py

    # Show plots of all results
    if not clargs.save_figures:
        plt.show()


if __name__ == '__main__':
    CLARGS = parse_args()
    exercise1(CLARGS)

