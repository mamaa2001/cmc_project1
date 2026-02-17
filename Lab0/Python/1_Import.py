#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Imports

This script introduces you to the useage of Imports in Python.  One of the most
powerful tool of any programming langauge is to be able to reuse code.  Python
allows this by setting up modules.  One can import existing libraries using the
import function.

"""

import farms_pylog as pylog
import datetime as dt
from math import sqrt, cos
import farms_pylog
from math import sqrt
import math
farms_pylog.info(3*'\t' + 20*'#' + ' IMPORTS ' + 20*'#' + 3*'\n')

# The command above will crash the program since pylog is not a standard
# python knows and no module has been imported that defines it.
# To be able to run the program comment the line and run again.

# A generic import of a default module named math

# Now you have access to all the functionality availble
# in the math module to be used in this function
print('Square root of 25 computed from math module : {}'.format(math.sqrt(25)))

# To import a specific function from a module
# Now you can avoid referencing that the sqrt function is from
# math module and directly use it.
print(
    'Square root of 25 computed from math module by'
    ' importing only sqrt function: {}'.format(sqrt(25))
)

# Import a user defined module
# Here we import farms_pylog, a module developed to display log messages for
# the exercises to facilitate logging

farms_pylog.info(
    'Module developed to display log messages for the exercises. You might'
    '\nnotice that the main difference with using the print() function is that'
    '\nit provides additional information. The format is the following:\n'
    '\n[PROCESS_NAME-PROCESS_ID] DATE TIME - MESSAGE_TYPE - FILENAME::LINE::FUNCTION:'
    '\n\nWe extensively make use of farms_pylog in order to make it easier for'
    '\nyou to find the source of messages in the exercises. Happy logging!')

# Importing multiple functions from the same module

# Defining an alias :
# Often having to reuse the actual name of module can be bothersome, e.g. due to
# the length of the name or due to duplicate/common names within the codes.  We
# can assign aliases to module names to avoid this problem.

pylog.info('Here we import the module datetime as dt.')
pylog.info(
    'Here is an example of how to use the module:\n'
    'dt.datetime.now().time() = %s',
    dt.datetime.now().time(),
)

# Getting to know the methods availble in a module
pylog.info(dir(math))

# Displaying variables with logging
a = math.pi
pylog.info(
    'In many cases, you might need to print variables, variables can easily be'
    '\nincorporated into strings using the str.format(...). Here is an example:'
    '\na = math.pi'
    '\n\'Value of a = {{}}\'.format(a) displays '
    '\'Value of a = {}\''.format(a) +
    '\nThis approach can be used for both print() and pylog()')

pylog.info(
    'Pylog is based on Python\'s logging module. An alternative approch for'
    '\nincorporating variables into strings is to use %%. Watch out, this'
    '\nwill not work with the print function.'
    '\na = math.pi'
    '\n%%s: %s (String)'
    '\n%%d: %d (Decimal)'
    '\n%%i: %i (Same as %%d)'
    '\n%%f: %f (Float)'
    '\n%%f: %.3f (Float, 3 decimals)'
    '\n%%f: %.50f (Float, 50 decimals)'
    '\n%%f: %10.2f (Float, fill with space to use 10 characters in total)'
    '\n%%f: %010.2f (Float, fill with \'0\' to use 10 characters in total)'
    '\n%%e: %e (Engineering notation)',
    a, a, a, a, a, a, a, a, a,
)

print(
    'Here is the equivalent to work with print():'
    '\na = math.pi'
    '\n%%s: %s (String)'
    '\n%%d: %d (Decimal)'
    '\n%%i: %i (Same as %%d)'
    '\n%%f: %f (Float)'
    '\n%%f: %.3f (Float, 3 decimals)'
    '\n%%f: %.50f (Float, 50 decimals)'
    '\n%%f: %10.2f (Float, fill with space to use 10 characters in total)'
    '\n%%f: %010.2f (Float, fill with \'0\' to use 10 characters in total)'
    '\n%%e: %e (Engineering notation)'
    % (a, a, a, a, a, a, a, a, a)
)

pylog.info(
    f'Finally, try out Python\'s f-strings, they might be very useful:'
    f'\n1 + 2 = {1 + 2}'
    f'\na = {a}'
    f'\nmath.pi = {math.pi}'
    f'\nmath.pi:f = {math.pi:f}'
    f'\nmath.pi:.3f = {math.pi:.3f}'
    f'\nmath.pi:10.3f = {math.pi:10.3f}'
    f'\nmath.pi:010.3f = {math.pi:010.3f}'
    f'\nmath.pi:.3e = {math.pi:.3e}'
)

