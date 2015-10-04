import numpy as np

c = np.dot(3, 4)
print(c)

c = np.dot([2j, 3j], [2j, 3j])
print(c)

a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
c = np.dot(a, b)
print(c)

# the next dot call will raise value error
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2], [1, 2, 3]]
c = np.dot(a, b)
print(c)