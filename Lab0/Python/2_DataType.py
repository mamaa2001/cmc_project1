#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Data types

This script explains the basic data types used in Python.
All the comman data types used in programming langauges are in Python too.

"""

# Import the farms_pylog module to display log messages
import farms_pylog as pylog

pylog.info('%s', 3*'\t' + 20*'#' + ' DATA TYPES ' + 20*'#' + 3*'\n')

pylog.warning(
    'In Python every element is treated as an Object.'
    '\nIncluding numbers and literals!!!'
)

# Different Data types
# Use 'type' method to identify the types
pylog.info('Data Type of 2 is : %s', type(2))  # int

pylog.info('Data Type of 2.0 is : %s', type(2.0))  # Float

pylog.info('Data Type of \'two\' is : %s', type('two'))  # String

# Strings in can be three types:
# 1. '' -> single line strings
# 2. '' -> single line strings
# 3. """ """ -> Multi line strings

pylog.warning('There is no separate char data type in Python.')

# Boolean
pylog.info('Data Type of keyword True is : %s', type(True))

# None type
pylog.info('Data Type of keyword None is : %s', type(None))

