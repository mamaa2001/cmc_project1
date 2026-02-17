#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Functions

This script introduces you to the usage of Functions in Python.
A function is defined in Python using the keyword def followed by function
name.
By default Python function returns none

"""

import farms_pylog as pylog  # import farms_pylog for log messages

pylog.info('%s', 3*'\t' + 20*'#' + ' FUNCTIONS ' + 20*'#' + 3*'\n')


# define a function with no arguments and no return values
def print_text():
    """Function docstring: A function which prints a text

    Additional information can be provided...

    """
    pylog.info('this is text')


# call the function
print_text()

pylog.info('A python file can contain more than one function!')


# define a function with one argument and no return values
def print_this(x):
    """Another function which prints information"""
    pylog.info(('Printing input \'x\'', x))


# call the function
print_this(3)       # prints 3
n = print_this(3)   # prints 3, but doesn't assign 3 to n
#   because the function has no return statement
pylog.warning(
    ' prints 3, but doesn\'t assign 3 to n'
    ' because the function has no return statement'
)


# define a function with one argument and one return value
def square_this(x):
    """Return the square of a number"""
    return x**2


# call the function
square_this(3)          # prints 9
var = square_this(3)    # assigns 9 to var, but does not print 9


# Define a function with two 'positional arguments' (no default values) and
# one 'keyword argument' (has a default value)
# Type hints can also be provided in Python, but are not enforced
def calc(a: float, b: float, op: str = 'add') -> float:
    """A function which operates on two values depending on the op kwarg"""
    if op == 'add':
        return a + b
    if op == 'sub':
        return a - b
    pylog.info('valid operations are add and sub')
    return None


pylog.info('Variables with default values should always be at the end!')
pylog.info(
    'Variables can be directly assigned a value irrespective of their'
    ' order if you use same name as in the function definition'
)


# call the function
calc(10, 4, op='add')   # returns 14
# also returns 14: unnamed arguments are inferred by position
calc(10, 4, 'add')
calc(10, 4)             # also returns 14: default for 'op' is 'add'
calc(10, 4, 'sub')      # returns 6
calc(10, 4, 'div')      # prints 'valid operations are add and sub'


# use 'pass' as a placeholder if you haven't written the function body
def stub():
    """A function which does nothing"""
    pass


# return two values from a single function
def min_max(numbers):
    """Returns (min, max)"""
    return min(numbers), max(numbers)


# return values can be assigned to a single variable as a tuple
nums = [1, 2, 3]
min_max_num = min_max(nums)         # min_max_num = (1, 3)

# return values can be assigned into multiple variables using tuple unpacking
min_num, max_num = min_max(nums)    # min_num = 1, max_num = 3


# Functions can variable inputs
# opertors *args and **kwargs provides two ways in python for a function
# to have arbitrary number of input arguments.
# The first method (*args) -> will contain the arguments as a tuple.
# The second method (**kwargs) -> will contain the arguments as
# key, value paired dictionary
# Both *args and **kwargs can be used together


# Use of *args
def function_with_args(*args):
    """Function accepts an arbitrary number of arguments"""
    # Print the number of arguments
    pylog.info(
        'Number of arguments in function_with_args are %s',
        len(args),
    )

    pylog.info(
        'Type of arguments in function_with_args is %s',
        type(args),
    )

    # Print all the args
    pylog.info(
        '\n'.join([
            f'{num} -> {arg}'
            for num, arg in enumerate(args)
        ])
    )


# Example with different of arguments using arg will be converted into
# a single tuple
function_with_args(1, 'string', 0.001, [1, 2, 3])


# Use of *kwargs
def function_with_kwargs(**kwargs):
    """Function accepts an arbitrary number of arguments and converts
    to a dictionary"""
    # Print the number of arguments
    pylog.info(
        'Number of arguments in function_with_kwargs are %s',
        len(kwargs),
    )

    pylog.info(
        'Type of arguments in function_with_kwargs is %s',
        type(kwargs),
    )

    # Print all the args
    pylog.info(
        '\n'.join([
            f'{num} {name} -> {arg}'
            for num, (name, arg) in enumerate(kwargs.items())
        ])
    )


# Example with different of arguments using kwarg will be converted into
# a single dictionary
function_with_kwargs(
    int_num=1,
    string='string',
    float_num=0.001,
    list_ints=[1, 2, 3],
)

