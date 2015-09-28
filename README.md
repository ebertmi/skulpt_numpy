# numpy module for skulpt
##Description
This is a partial port of some numpy functions for skulpt. Mainly, focused on
```ndarray``` and its respective functions. See below for the currently supported functions.

## News
#### Latest
- improved internal attribute handling
- improved internal buffer handling
- added ```ndarray.T``` attribute
- added ```__getattr__```
- added ```__setattr__```
- added ```numpy.vdot```

Example:
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a)
a.shape = (4,)
print(a)
a.shape = (1, 2, 2)
print(a.T)
a.shape = 4
print(a)

a = np.array([[1j, 2j], [3j, 4j]])
b = [1, 2, 3, 4]
c = np.vdot(a, b)

print(c)
```

##Supported

###ndarray
Currently all attributes are readonly. Though
- [x] ```__str__```
- [x] ```__repr__```
- [x] ```__len__```
- [x] ```shape```
- [x] ```ndim```
- [x] ```data```
- [x] ```dtype```
- [x] ```size```
- [x] ```tolist```
- [x] ```fill```
- [x] ```[], with slices```
- [x] ```operators: +, -, /, *, **, %, neg, pos, xor, shift```
- [x] ```reshape```
- [x] ```copy```
- [x] ```__iter__```

###functions
- [x] linspace
- [x] array
- [x] ones
- [x] zeros
- [x] full
- [x] empty
- [x] arange
- [x] asarray

###trigonomeric
Some of the functions rely on math.js, but you can use the module without it. All
functions that require the library are going to throw an python exception if they
cannot call mathjs.


- [x] dot (mathjs)
- [x] sin
- [x] cos
- [x] tan
- [x] sinh (mathjs)
- [x] cosh (mathjs)
- [x] tanh (mathjs)
- [x] arctan
- [x] arcsin
- [x] arccos

###Examples
```python
import numpy as np

a = [[1, 2], [3, 4]]

b = np.array(a, ndmin=3, dtype=float)
print "np.array(a, ndmin=3a, dtype=float)"

c = np.ones(b.shape)
print "np.ones(b.shape): %s" % (c,)
d = np.zeros(b.shape)
print "np.zeros(b.shape): %s" % (d,)
print "__str__: %s" % b
print "__repr__: %r" % b
print "__len__: %s" % len(b)
print "shape %s" % (b.shape,)
print "ndim %s" % b.ndmin
print "data: %s"  % (b.data,)
print "dtype: %s" % b.dtype
print "size %s" % b.size
print "b.tolist %s" % (b.tolist(),)
b.fill(9)
print "b.fill(9): %s" % (b,)
b[0, 0, 0] = 2
print "b[0, 0, 0] = 2: %s" % (b,)

print ""
print "np.full((2,2), 2.0)"
c = np.full((2,2), 2.0, int)

print "===================="
print "     operations"
print "===================="
print "c = %s" % (c,)
print "c + 2 = %s" % (c + 2,)
print "c - 2 = %s" % (c - 2,)
print "c * 2 = %s" % (c * 2,)
print "c / 2 = %s" % (c / 2,)
print "c ** 2 = %s" % (c ** 2,)
print "+c = %s" % (+c,)
print "-c = %s" % (-c,)

print "===================="
print "   trigonometric    "
print "===================="
c = np.full((2,2), 0, int)
print "c = %s" % (c,)
print "np.sin(c) = %s" % (np.sin(c),)
print "np.cos(c) = %s" % (np.cos(c),)
print "np.tan(c) = %s" % (np.tan(c),)
print "np.arcsin(c) = %s" % (np.arcsin(c),)
print "np.arccos(c) = %s" % (np.arccos(c),)
print "np.arctan(c) = %s" % (np.arctan(c),)
print "np.sinh(c) = %s" % (np.sinh(c),)
print "np.cosh(c) = %s" % (np.cosh(c),)
print "np.tanh(c) = %s" % (np.tanh(c),)
print "np.sin([0,1]) = %s" % (np.sin([0,1]),)
print "np.sin((0,1)) = %s" % (np.sin((0,1)),)

print "===================="
print "      various       "
print "===================="
ar = np.arange(3.0)
print "np.arange(3.0): %s, dtype: %s" % (ar, ar.dtype)
ar = np.arange(3)
print "np.arange(3): %s, dtype: %s" % (ar, ar.dtype)
ar = np.arange(3,7)
print "np.arange(3,7): %s, dtype: %s" % (ar, ar.dtype)
ar = np.arange(3,7, 2)
print "np.arange(3,7, 2): %s, dtype: %s" % (ar, ar.dtype)
ar = np.linspace(2.0, 3.0, num=5)
print "np.linspace(2.0, 3.0, num=5): %s" % (ar,)
ar = np.linspace(2.0, 3.0, num=5, endpoint=False)
print "np.linspace(2.0, 3.0, num=5, endpoint=False): %s" % (ar,)
ar = np.linspace(2.0, 3.0, num=5, retstep=True)
print "np.linspace(2.0, 3.0, num=5, retstep=True): %s" % (ar,)
```
