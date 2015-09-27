https://github.com/numpy/numpy/blob/master/numpy/random/__init__.py
# Implementation of the numpy.random module (basic functionality)
## RandomState
The ```RandomState``` class is working and can be already used to generate a new instance with a given seed.

## Function Mappings
### rand(d0, d1, ..., dn)
Numpy.random module maps ```numpy.random.rand``` to an internal instance of ```RandomState```.

You can already call the function like this:
```python
import numpy.random as random

ca = random.RandomState(seed=None)
a = 2.5 * random.rand(2, 4) + 3

print(a)
```

### seed
Or you can use any array-like data structure for seeding:
```python
import numpy.random as random

ca = random.RandomState(seed=(1, 2, 3, 4))
a = 2.5 * ca.rand(2, 4) + 3

print(a)
```

### Milestones
Support the following example:
```python
import numpy as np
import numpy.random as random

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*random.random((3,1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print("Output After Training:")
print(l1)
```

Currently this fails due to missing ndarray.transpose and ndarray.T method!