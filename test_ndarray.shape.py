import numpy as np
a = np.array([1, 2, 3, 4])
a.shape = (1, -1)
a.shape = (1, -1, 1)
a.shape = -1
print(a)

try:
    a.shape = None
except ValueError:
    print('total size of a new array must be unchanged')

try:
    a.shape = {}
except TypeError:
    print('shape requires an int or sequence')

try:
    a.shape = (1, '-1')
except TypeError:
    print('dimension must be of type int')

try:
    a.shape = (-1, -1)
except ValueError:
    print('there must be only one unknown dimension')
