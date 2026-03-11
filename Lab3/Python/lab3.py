#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Lab 3 """

from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt

import farms_pylog as pylog
from cmcpack import parse_args, integrate_multiple, DEFAULT
from cmcpack.results import MultipleResultsODE

# pylint: disable=invalid-name


class LeakyIntegratorParameters:
    """ Leaky-integrator neuron parameters """

    def __init__(
            self,
            tau: np.array,
            D: float,
            b: np.array,
            w: np.array,
            exp: Callable = np.exp,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.tau: np.array = np.array(tau)  # Time constant
        self.D: float = np.array(D)
        self.b: np.array = np.array(b)
        self.w: np.array = np.array(w)  # Weights
        self.exp: Callable = exp  # Exponential

    def __str__(self) -> str:
        """ String used when printing instantiated object """
        return self.msg()

    def list(self) -> List:
        """ Return list of parameters """
        return self.tau, self.D, self.b, self.w, self.exp

    def msg(self) -> str:
        """ Parameters information message """
        return (
            'Leaky integrator parameters:'
            f'\nTau: {self.tau}'
            f'\nD:   {self.D}'
            f'\nb:   {self.b}'
            f'\nw:   {self.w}'
            f'\nExp: {self.exp}'
        )


def two_li_ode(
        y: np.array,
        t: float,
        params: LeakyIntegratorParameters,
) -> np.array:
    """ Derivative function of a network of 2 leaky integrator neurons

    y is the vector of membrane potentials (variable m in lecture equations)
    yd the derivative of the vector of membrane potentials
    """
    # pylint: disable=unused-argument
    # Extract parameters
    tau, D, b, w, exp = params.list()

    # Update the firing rates:
    x = [0, 0]

    # IMPLEMENT THE DIFFERENTIAL EQUATION FOR THE MEMBRANE POTENTIAL
    # Compute the dentritic sums for both neurons
    dend_sum = [0, 0]

    # Compute the membrane potential derivative:
    yd = [0, 0]
    # pylog.debug(f'x: {x}\ndend_sum: {dend_sum}\nyd: {yd}')
    return yd


def two_coupled_li_neurons(
        y_0: np.array,
        t_max: float,
        dt: float,
        params: LeakyIntegratorParameters,
        figure: str = 'Phase',
) -> MultipleResultsODE:
    """ Two mutually coupled leaky-integrator neurons with self connections """
    res = integrate_multiple(
        two_li_ode,
        y_0,
        np.arange(0, t_max, dt),
        args=(params,)
    )
    labels = ['Neuron 1', 'Neuron 2']
    res.plot_state(figure+'_state', label=False, subs_labels=labels)
    res.plot_phase(figure+'_phase', scale=0.05, label=labels)
    return res


def exercise5():
    """ Lab 3 - Exrecise 5 """
    # Fixed parameters of the neural network
    tau = [0.05, 0.05]
    D = 1

    # Additional parameters
    b = [0, 0]
    w = [[1, 0], [0, 1]]

    # All system parameters packed in object for integration
    params = LeakyIntegratorParameters(tau, D, b, w)

    # Initial conditions
    #  SET THE INITIAL CONDITIONS
    y_0 = [[0, 0]]  # Values of the membrane potentials of the two neurons

    # Integration time parameters
    # Force integration to return values at small steps for
    # better detecting Poincare maps crossings
    dt = 1e-4
    t_max = 30  # Set total simulation time

    # Integration (make sure to implement)
    pylog.warning(
        'Uncomment next line to run integration'
        ' after implementing two_li_ode()'
    )
    # two_coupled_li_neurons(y_0, t_max, dt, params, 'Case1')

    # Two stable fixed points and one saddle node
    pylog.warning('Implement two stable fixed points and one saddle node')

    # Limit cycle
    pylog.warning('Implement limit cycle')
    pylog.warning('Implement Poincare analysis of limit cycle')

    # Limit cycle (small), one stable fixed point and one saddle node
    pylog.warning(
        'Implement a system with:'
        '\n- One limit cycle (small)'
        '\n- One stable fixed point'
        '\n- One saddle node'
    )
    pylog.warning('Implement Poincare analysis of limit cycle')

    if DEFAULT['save_figures'] is False:
        plt.show()


def main():
    """ Lab 3 exercise """
    pylog.info('Runnig exercise 5')
    exercise5()


if __name__ == '__main__':
    parse_args()
    main()

