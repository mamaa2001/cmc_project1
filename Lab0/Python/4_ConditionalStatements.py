#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Conditional statements

This script discussess the different conditional statements in Python.

"""

import farms_pylog as pylog  # Import farms_pylog module for log messages

pylog.info('%s', 3*'\t'+20*'#'+' CONDITIONAL STATEMENTS '+20*'#'+3*'\n')


# Before dwelling into the conditional statements, first let us look at the
# Comparison and Boolean Operators availble

# Comparison and Boolean Operations

x = 5  # Assignment Statement
# Remember that Python treats every element in the program as an Object

# Comparisons
pylog.info('x > 3 : %s', x > 3)  # Greater than

pylog.info('x < 3 : %s', x < 3)  # Lesser than

pylog.info('x >= 3 : %s', x >= 3)  # Greater than or equal to

pylog.info('x != 3 : %s', x != 3)  # Not equal to

pylog.info('x == 3 : %s', x == 3)  # Equal to

pylog.info('1 < 2 < 3 : %s', 1 < 2 < 3)  # chained comparison

# Boolean operations

pylog.info('5 > 3 and 6 < 3 : %s', 5 > 3 and 6 < 3)  # and operator

pylog.info('(5 > 3) or (6 > 3) : %s', 5 > 3 or 6 > 3)  # or operator

pylog.info('not False : %s', not False)  # not operator

# Evaluation order : not, and, or

pylog.info('False or not False and True : %s', False or not False and True)

pylog.info('Try to implement >= operator with \'or\' operator')

# Conditional Statements

# if statement
if x > 0:
    pylog.info('Positive')

# if/else statement
if x > 0:
    pylog.info('Positive')
else:
    pylog.info('Zero or Negative')

# if/elif/else statement
if x > 0:
    pylog.info('Positive')
elif x == 0:
    pylog.info('Zero')
else:
    pylog.info('Negative')

# Ternary operator
check = 'Positive' if x > 0 else 'Zero or Negative'
pylog.info('Ternary operator : %s', check)

