#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Data containers

This script explains the different ways in Python to store and group data.
The three major ways in which you store data are
1. Lists
2. Tuple
3. Dictionaries

"""

import farms_pylog as pylog  # Import pylog to log messages

# Lists
# properties: ordered, iterable, mutable, can contain multiple data types
pylog.info('%s', 3*'\t' + 20*'#' + ' LISTS ' + 20*'#' + 3*'\n')

# create an empty list (two ways)
empty_list = []
empty_list = list()

# create a list
course = ['computational', 'motor', 'control']

# examine a list
pylog.info('Examine the course list : %s', course)  # Show course list
pylog.info(
    'Examine the first element of the course : %s',
    course[0],  # print element 0 ('homer')
)

pylog.warning('Indexing in Python starts from 0 and not 1')

pylog.info(
    'Print the length of the list : %s',
    len(course),  # returns the length (3)
)

# modify a list (does not return the list)
course.append('I')                 # append element to end
pylog.info('.append : %s', course)

course.extend(['love', 'Python'])  # append multiple elements to end
pylog.info('.extend : %s', course)

# insert element at index 0 (shifts everything right)
course.insert(0, '2026 : ')
pylog.info('.insert : %s', course)

# search for first instance and remove it
course.remove('Python')
pylog.info('.remove : %s', course)

course.pop(0)                         # remove element 0 and return it
pylog.info('.pop : %s', course)

del course[0]                         # remove element 0 (does not return it)
pylog.info('del element [0] : %s', course)

course[0] = '?????????????'                  # replace element 0
pylog.info('replace element [0] : %s', course)

# concatenate lists
students = course + ['student1', 'student2', 'student3']
pylog.warning('This method is slower than \'.extend\' method')

# find elements in a list
course.count('motor')      # counts the number of instances
pylog.info('Number of instances of motor in the list : %s',
           course.count('motor'))

# list slicing [start:end:step]
pylog.info('Indexing or slicing in Python [start:end:step]')
weekdays = ['mon', 'tues', 'wed', 'thurs', 'fri']
pylog.info('weekdays list: %s', weekdays)
w1 = weekdays[0]         # element 0
pylog.info('weekdays[0] list: %s', w1)
w2 = weekdays[0:3]       # elements 0, 1, 2
pylog.info('weekdays[0:3] list: %s', w2)
w3 = weekdays[:3]        # elements 0, 1, 2
pylog.info('weekdays[:3] list: %s', w3)
w4 = weekdays[3:]        # elements 3, 4
pylog.info('weekdays[3:] list: %s', w4)
w5 = weekdays[-1]        # last element (element 4)
pylog.info('weekdays[-1] list: %s', w5)
w6 = weekdays[::2]       # every 2nd element (0, 2, 4)
pylog.info('weekdays[::2] list: %s', w6)
w7 = weekdays[::-1]      # backwards (4, 3, 2, 1, 0)
pylog.info('weekdays[::-1] list: %s', w7)

pylog.info('Try to see if there are other methods to reverse a list')

# sort a list in place (modifies but does not return the list)
pylog.info('course variable contains %s', course)
course.sort()
pylog.info('course variable after sort %s', course)
course.sort(reverse=True)     # sort in
pylog.info('course variable after reverse sort %s', course)


# create a second reference to the same list
pylog.info('Copying lists only creates a new reference and not a new list')

same_course = course
pylog.info('same_course = course => same_course = %s (reference)', same_course)
same_course[0] = 0         # modifies both 'course' and 'same_course'
pylog.info('same_course[0] = 0 => same_course = %s', same_course)

pylog.info(
    'Element 0 in course : %s\nElement 0 in same_course : %s',
    course[0],
    same_course[0],
)
pylog.info('same_course and course have same values!!')

# copy a list (two ways)
pylog.info('To make a new copy of a list use slicing methods')
new_course = course[:]
pylog.info('new_course = course[:] => new_course = %s (copy)', new_course)
new_course[0] = 0         # modifies both 'course' and 'same_course'
pylog.info('new_course[0] = 0 => new_course = %s', new_course)

pylog.info(
    'Element 0 in course : %s\nElement 0 in same_course : %s',
    course[0],
    new_course[0],
)

# examine objects
# returns True (checks whether they are the same object)
c1 = course is same_course
pylog.info('check if course & same_course are same with `is` keyword : %s', c1)
c2 = course is new_course          # returns False
pylog.info('check if course & new_course are same with `is` keyword: %s', c2)
# returns True (checks whether they have the same contents)
c3 = course == same_course
pylog.info('check if course & same_course are same with == : %s', c3)
c4 = course == new_course          # returns True
pylog.info('check if course & new_course are same with == : %s', c4)


# TUPLES
print(3*'\t' + 20*'#' + ' TUPLES ' + 20*'#' + 3*'\n')

pylog.info('Tuples are like lists but don\'t change in size')

# Create a tuple
pylog.info('Tuples are created with \'(...)\' or using the keyword tuple')

digits = (0, 1, 'two')  # Create a tuple directly
pylog.info('digits = (0, 1, \'two\') => digits = %s', digits)
digits = tuple([0, 1, 'two'])  # Create a tuple from list
pylog.info('digits = tuple([0, 1, \'two\']) => digits = %s', digits)
pylog.warning('A trailing comma is required to indicate its a tuple')
zero_tuple = (0,)
pylog.warning('zero_tuple = (0,) => type(zeros_tuple) = %s', type(zero_tuple))
zero_num = (0)
pylog.warning('zero_num = (0) => type(zeros_num) = %s', type(zero_num))
# examine a tuple

pylog.info('Examine tuples with \'digits\'')
d1 = digits[2]  # returns 'two'
pylog.info('digits[2] = %s', d1)
d2 = len(digits)  # returns 3
pylog.info('len(digits) = %s', d2)
d3 = digits.count(0)  # counts the number of instances of that value
pylog.info('digits.count(0) = %s', d3)
d4 = digits.index(1)  # returns the index of the first instance of that value
pylog.info('digits.index(1) = %s', d4)

# elements of a tuple cannot be modified
try:
    digits[2] = 2       # throws an error
except BaseException:
    pylog.error(
        'digits[2] = 2 => Error,'
        ' Tuples cannot be edited once initialized',
    )

# concatenate tuples
digits = digits + (3, 4)
pylog.info('concatenate tuples, digits + (3, 4)  = %s', digits)

# create a single tuple with elements repeated (also works with lists)
repeat = (3, 4) * 2          # returns (3, 4, 3, 4)
pylog.info('repeat tuples elements,  (3, 4)*2  = %s', repeat)

# sort a list of tuples
tens = [(20, 60), (10, 40), (20, 30)]
pylog.info('original variable tens = %s', tens)
# The sorted() function sorts by first element in tuple, then second element
tens_sorted = sorted(tens)  # returns [(10, 40), (20, 30), (20, 60)]
pylog.info('sorted(tens) = %s', tens_sorted)

# tuple unpacking
bart = ('male', 10, 'simpson')  # create a tuple
pylog.info('bart = %s', bart)
(sex, age, surname) = bart      # assign three values at once
pylog.info(
    'unpack tuple with (sex, age, surname) = bart =>'
    ' sex = %s,'
    ' age = %s,'
    ' surname = %s',
    sex, age, surname,
)


# DICTIONARY
print(3*'\t' + 20*'#' + ' DICTIONARY ' + 20*'#' + 3*'\n')

# properties: unordered, iterable, mutable, can contain multiple data types
# made of key-value pairs
# keys must be unique, and can be strings, numbers, or tuples
# values can be any type

# create an empty dictionary (two ways)
pylog.info(
    'Dictionaries are created with \'{...}\' or using the keyword dict')

empty_dict = {}
empty_dict = dict()

# create a dictionary (two ways)
family = {'dad': 'homer', 'mom': 'marge', 'size': 6}
pylog.info('initialize using family = %s => family = %s', '{..}', family)
family = dict(dad='homer', mom='marge', size=6)
pylog.info('initialize using family = dict(...) => family = %s', family)

# convert a list of tuples into a dictionary
list_of_tuples = [('dad', 'homer'), ('mom', 'marge'), ('size', 6)]
family = dict(list_of_tuples)
pylog.info('dict(list_of_tuples) = %s', family)

# examine a dictionary
v1 = family['dad']       # returns 'homer'
pylog.info('family[\'dad\'] = %s', v1)
v2 = len(family)         # returns 3
pylog.info('len(family) = %s', v2)
v3 = 'mom' in family     # returns True
pylog.info('\'mom\' in family? %s', v3)
v4 = 'marge' in family   # returns False (only checks keys)
pylog.info('\'marge\' in family? %s', v4)

# returns a list (Python 2) or an iterable view (Python 3)
l1 = list(family.keys())       # keys: ['dad', 'mom', 'size']
l2 = list(family.values())     # values: ['homer', 'marge', 6]
# key-value pairs: [('dad', 'homer'), ('mom', 'marge'), ('size', 6)]
l3 = list(family.items())

# modify a dictionary (does not return the dictionary)
pylog.info('family = %s', family)
family['cat'] = 'snowball'              # add a new entry
pylog.info('adding cat, family[\'cat\'] = %s', family['cat'])
family['cat'] = 'snowball ii'           # edit an existing entry
pylog.info('renaming cat, family[\'cat\'] = %s', family['cat'])
del family['cat']                       # delete an entry
pylog.info('family after deleting \'cat\', family =  %s', family)
family['kids'] = ['bart', 'lisa']       # dictionary value can be a list
pylog.info('family after adding \'kids\', family = %s', family)

# remove an entry and return the value ('homer')
p1 = family.pop('dad')
pylog.info('family after pop (\'dad\'), family = %s', family)
pylog.info('return of pop function = %s', p1)
family.update({'baby': 'maggie', 'grandpa': 'abe'})   # add multiple entries
pylog.info('family after update function, family = %s', family)

# access values more safely with 'get'
f1 = family['mom']                       # returns 'marge'
pylog.info('access using [\'key\'], family[\'mom\']  = %s', f1)
f2 = family.get('mom')                   # equivalent
pylog.info('access using get, family.get(\'mom\') = %s', f2)

try:
    # throws an error since the key does not exist
    family['grandma']
except BaseException:
    pylog.error(
        'Key grandma does not exist.'
        ' Try using get method instead to check the keys'
    )

family.get('grandma')               # returns None instead

family.get('grandma', 'not found')  # returns 'not found' (the default)

# access a list element within a dictionary
f3 = family['kids'][0]                   # returns 'bart'
pylog.info('access %s using family[\'kids\'][0]', f3)
family['kids'].remove('lisa')            # removes 'lisa'
pylog.info('family[\'kids\'].remove(\'lisa\')  => family = %s', family)

# String substitution using a dictionary
pylog.info(
    'String substitution using a dictionary:'
    '\nfamily = %s'
    '\n\'youngest child is %%(baby)s\' %% family'
    ' => %s',
    family,
    'youngest child is %(baby)s' % family  # returns 'youngest child is maggie'
)

