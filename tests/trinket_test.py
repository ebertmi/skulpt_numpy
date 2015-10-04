import numpy as np
import random as rd
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
rd.seed(1)
syn0_2 = []
syn0_2.append(rd.random())
syn0_2.append(rd.random())
syn0_2.append(rd.random())
print(random.random((3)))
print(syn0_2)
# initialize weights randomly with mean 0
syn0 = 2*random.random((3,1)) - 1
print(syn0)

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    new_w = np.dot(l0.T, l1_delta)
    syn0 = syn0 + new_w

print("Output After Training:")
print(l1)