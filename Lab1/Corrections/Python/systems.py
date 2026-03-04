""" Systems related codes for exercise 2 (Only for corrections) """

from typing import Any
import numpy as np
import farms_pylog as pylog

try:
    import sympy
    IMPORT_SYMPY = True
except ImportError:
    IMPORT_SYMPY = False

# pylint: disable=invalid-name


def fixed_point(A:np.array):
    """ Compute fixed point """
    x = sympy.symbols([f'x{i}' for i in range(2)])
    sol = sympy.solve(np.dot(A, x), x)
    x0 = sol[x[0]], sol[x[1]]
    pylog.info('Fixed point: %s', x0)


def eigen(A:np.array):
    """ Compute eigenvectors and eigenvalues """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    pylog.info('Eigenvalues: %s', eigenvalues)
    pylog.info('Eigenvectors:\n%s', eigenvectors)
    return eigenvalues, eigenvectors


def system_analysis(A:np.array):
    """ Exercise 2.a - Analyse system """
    if IMPORT_SYMPY:
        fixed_point(A)
        eigen(A)