#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Math

This script discussess the basic math operations used in Python.

"""

# Only necessary in Python 2

import farms_pylog as pylog  # Import farms_pylog module for log messages

pylog.info('%s', 3*'\t' + 20*'#' + ' MATH ' + 20*'#' + 3*'\n')

# Basic operations
pylog.info('Adding 132 + 123 : %s', 132 + 123)  # Add

pylog.info('Subrtacting 43 - 23 : %s', 43 - 23)  # Subtract

pylog.info('Multiply 65 * 87 : %s', 65 * 87)  # Multiply

pylog.info('Exponent 65**4 : %s', 65**4)  # Exponent

pylog.info('Modulo 5 %% 4 : %s', 5 % 4)  # Modulo

pylog.info('Division 10 / 4 : %s', 10/4)  # Division

pylog.info('Division 10.0 / 4 : %s', 10.0/4)  # True division (Python2)

pylog.info('Integer division 10 // 4 : %s', 10//4)  # Integer division

