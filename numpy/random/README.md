README.md
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