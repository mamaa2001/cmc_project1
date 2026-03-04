""" Lab 2 """

import farms_pylog as pylog
from exercise1 import exercise1
from exercise2 import exercise2


def main():
    """Main function that runs all the exercises."""
    pylog.info('Implementing Lab 2 : Exercise 1')
    exercise1()
    exercise2()


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    main()

