""" Lab 1 - Exercise 2 """

from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from cmcpack import integrate, parse_args, plot
from systems import system_analysis

# pylint: disable=invalid-name


def ode(
        x: np.array,
        _: float = None,
        A: np.array = np.eye(2)
) -> np.array:
    """ System x_dot = A*x """
    return np.dot(A, x)


def integration(
        x0: np.array,
        time: np.array,
        A: np.array,
        name: str,
        **kwargs,
):
    """ System integration """
    labels = kwargs.pop('label', [f'State {i}' for i in range(2)])
    sys_int = integrate(ode, x0, time, args=(A,))
    sys_int.plot_state(f'{name}_state', labels)
    sys_int.plot_phase(f'{name}_phase')


def exercise2(clargs):
    """ Exercise 2 """
    pylog.info('Running exercise 2')

    # System definition
    A = np.array([[1, 4], [-4, -2]])
    system_analysis(A)  # Optional
    time_total = 10
    time_step = 0.01
    x0, time = [0, 1], np.arange(0, time_total, time_step)

    # Normal run
    pylog.info('Running system integration')
    integration(x0, time, A, 'system_integration')

    # Stable point (Optional)
    pylog.info('Running stable point integration')
    x0 = [0, 0]
    integration(x0, time, A, 'stable')

    # Periodic
    pylog.info('Running periodic system integration')
    A = np.array([[2, 4], [-4, -2]])
    x0 = [1, 0]
    integration(x0, time, A, 'periodic')

    # Saddle
    pylog.info('Running saddle node system integration')
    A = np.array([[1, 1], [4, -2]])
    initial_points = -0.5 + 1*np.random.randn(100, 2)

    figname = 'saddle node phase diagram'
    plt.figure(figname)

    for i, x0 in enumerate(initial_points):
        sys_int = integrate(ode, x0, time, args=(A,))
        for s in [sys_int.state]:
            plot.plot_phase_trajectory(s, 0.4)

        if i == 0:
            # Quiver
            quiver_range = [[-2, 2], [-2, 2]]
            plot.plot_phase_quiver(sys_int.ode, sys_int.args, quiver_range)

    # Eigenvectors
    U = [1/np.sqrt(2), 1/np.sqrt(17)]
    V = [1/np.sqrt(2), -4/np.sqrt(17)]
    colors = ('r', 'g')
    for i in range(2):
        plt.plot(
            [-U[i]*5, U[i]*5],
            [-V[i]*5, V[i]*5],
            colors[i],
            linewidth=2,
            label=f'Eigenvector {i}',
        )
    plt.plot(
        [0], [0],
        'ko',
        label='Saddle node',
    )

    plt.xlabel('State 0')
    plt.ylabel('State 1')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plot.save_figure(figname)

    # Plot
    if not clargs.save_figures:
        plt.show()


if __name__ == '__main__':
    CLARGS = parse_args()
    exercise2(CLARGS)