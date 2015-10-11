import numpy as np
import numpy.random as random
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1] ])

l0 = X.transpose()
#print(l0)
a = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]])

print(a[0:4])
print(l0[0:4])

b = np.array([[-0.0528915261329], [-0.0841850124442], [0.138112061308], [0.148097986397]])
c = np.dot(a, b)
d = np.dot(l0, b)
print(c)
print(d)