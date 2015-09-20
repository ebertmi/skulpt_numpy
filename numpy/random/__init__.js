/* eslint-disable camelcase, no-eq-null, comma-dangle, no-underscore-dangle, strict, new-cap, no-var, vars-on-top, no-param-reassign, func-names */
/* ****************************************************************************/
/*                                  RandomKit                                 */
/*                                                                            */
/* ****************************************************************************/

var RK_STATE_LEN = 624;

function createRkState(key, pos, has_gauss, has_binominal, psave, nsave, r, q, fm, m, p1, xm, xl, xr, laml, lamr, p2, p3, p4) {
    return {
        key: key || new Array(RK_STATE_LEN),
        pos: pos || null,
        gauss: null,
        has_gauss: has_gauss || null, /* !=0 gauss contains a gaussion deviate */

        /* The rk_state structure has been extended to store the following
         * information for the binomial generator. If the input values of n or p
         * are different than nsave and psave, then the other parameters will be
         * recomputed. RTK 2005-09-02 */

        has_binominal: has_binominal || null, /* !=0 following parameters initialized for binomial */

        psave: psave || null,
        nsave: nsave || null,
        r: r || null,
        q: q || null,
        fm: fm || null,
        m: m || null,
        p1: p1 || null,
        xm: xm || null,
        xl: xl || null,
        xr: xr || null,
        laml: laml || null,
        lamr: lamr || null,
        p2: p2 || null,
        p3: p3 || null,
        p4: p4 || null,
    };
}

var rk_error = {
    RK_NOERR: 'RK_NOERR', /* no error */
    RK_ENODEV: 'RK_ENODEV', /* no RK_DEV_RANDOM device */
    RK_ERR_MAX: 'RK_ERR_MAX',
};

var rk_strerror = [
    'no error',
    'random device unavailable',
];

var RK_MAX = 0xFFFFFFFF;

var rk_hash_uint;
if (typeof Uint32Array === undefined) {
    rk_hash_uint = [0];
} else {
    rk_hash_uint = new Uint32Array(1);
}

/* Thomas Wang 32 bits integer hash function */
function rk_hash(key) {
    rk_hash_uint[0] = key | 0;
    rk_hash_uint[0] += ~(rk_hash_uint[0] << 15);
    rk_hash_uint[0] ^= (rk_hash_uint[0] >>> 10);
    rk_hash_uint[0] += (rk_hash_uint[0] << 3);
    rk_hash_uint[0] ^= (rk_hash_uint[0] >>> 6);
    rk_hash_uint[0] += ~(rk_hash_uint[0] << 11);
    rk_hash_uint[0] ^= (rk_hash_uint[0] >>> 16);
    return rk_hash_uint[0] >>> 0;
}

/*
 * Initialize the RNG state using the given seed.
 */
function rk_seed(seed, state) {
    var pos;
    seed &= 0xffffffff;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < RK_STATE_LEN; pos++) {
        state.key[pos] = seed;
        seed = (1812433253 * (seed ^ (seed >>> 30)) + pos + 1) & 0xffffffff;
    }
    state.pos = RK_STATE_LEN;
    state.gauss = 0;
    state.has_gauss = 0;
    state.has_binomial = 0;
}

/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */
function rk_randomseed(state) {
    var i;
    var tv;

    if (rk_devfill(state.key, 4, 0) === rk_error.RK_NOERR) {
        /* ensures non-zero key */
        state.key[0] |= 0x80000000;
        state.pos = RK_STATE_LEN;
        state.gauss = 0;
        state.has_gauss = 0;
        state.has_binomial = 0;

        for (i = 0; i < 624; i++) {
            state.key[i] &= 0xffffffff;
        }
        return rk_error.RK_NOERR;
    }

    tv = new Date();

    // we do not have access to the cpu clock!
    rk_seed(rk_hash(tv.getTime()) ^ rk_hash(tv.getMilliseconds()), state);

    return rk_error.RK_ENODEV;
}

/* Magic Mersenne Twister constants */
var N = 624;
var M = 397;
var MATRIX_A = 0x9908b0df;
var UPPER_MASK = 0x80000000;
var LOWER_MASK = 0x7fffffff;

/*
 * Returns a random unsigned long between 0 and RK_MAX inclusive
 * Slightly optimised reference implementation of the Mersenne Twister
 */
function rk_random(state) {
    var y;

    if (state.pos === RK_STATE_LEN) {
        var i;

        for (i = 0; i < N - M; i++) {
            y = (state.key[i] & UPPER_MASK) | (state.key[i + 1] & LOWER_MASK);
            state.key[i] = state.key[i + M] ^ (y >>> 1) ^ (-(y & 1) & MATRIX_A);
        }

        for (; i < N - 1; i++) {
            y = (state.key[i] & UPPER_MASK) | (state.key[i + 1] & LOWER_MASK);
            state.key[i] = state.key[i + (M - N)] ^ (y >>> 1) ^ (-(y & 1) & MATRIX_A);
        }
        y = (state.key[N - 1] & UPPER_MASK) | (state.key[0] & LOWER_MASK);
        state.key[N - 1] = state.key[M - 1] ^ (y >>> 1) ^ (-(y & 1) & MATRIX_A);

        state.pos = 0;
    }

    y = state.key[state.pos++];

    /* Tempering */
    y ^= (y >>> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >>> 18);

    // some javascript specifics
    return y >>> 0;
}

/*
 * Returns a random unsigned long between 0 and ULONG_MAX inclusive
 */
function rk_ulong(state) {
    return rk_random(state);
}

/*
 * Returns a random long between 0 and LONG_MAX inclusive
 */
function rk_long(state) {
    return rk_ulong(state) >>> 1;
}

/*
 * Returns a random unsigned long between 0 and max inclusive.
 */
function rk_interval(max, state) {
    var mask = max;
    var value;

    if (max === 0) {
        return 0;
    }

    /* Smallest bit mask >= mask */
    mask |= mask >>> 1;
    mask |= mask >>> 2;
    mask |= mask >>> 4;
    mask |= mask >>> 8;
    mask |= mask >>> 16;
    // mask |= mask >>> 32;

    if (max <= 0xffffffff) {
        while ((value = (rk_random(state) & mask)) > max) {
            // empty block
        }
    } else {
        while ((value = (rk_ulong(state) & mask)) > max) {
            // empty block
        }
    }

    return value;
}

/*
 * Returns a random double between 0.0 and 1.0, 1.0 excluded.
 */
function rk_double(state) {
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    var a = rk_random(state) >>> 5;
    var b = rk_random(state) >>> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

/*
 * fill the buffer with size random bytes
 * call with var buffer = new ArrayBuffer(16)
 * TODO: test this o.O
 */
function rk_fill(buffer, size, state) {
    var r;
    var buf = new Int32Array(buffer);
    var i = 0;

    for (; size >= 4; size -= 4) {
        r = rk_random(state);
        buf[i++] = r & 0xFF;
        buf[i++] = (r >>> 8) & 0xFF;
        buf[i++] = (r >>> 16) & 0xFF;
        buf[i++] = (r >>> 24) & 0xFF;
    }

    if (!size) {
        return;
    }

    r = rk_random(state);
    for (; size; r >>>= 8, size--) {
        buf[i++] = r & 0xFF;
    }
}

/*
 * fill the buffer with randombytes from the random device
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 * On Unix, if strong is defined, RK_DEV_RANDOM is used. If not, RK_DEV_URANDOM
 * is used instead. This parameter has no effect on Windows.
 * Warning: on most unixes RK_DEV_RANDOM will wait for enough entropy to answer
 * which can take a very long time on quiet systems.
 */
function rk_devfill(buffer, size, strong) {
    /* we have not access to the underlying random devices! */
    return rk_error.RK_ENODEV;
}

/*
 * fill the buffer using rk_devfill if the random device is available and using
 * rk_fill if is is not
 * parameters have the same meaning as rk_fill and rk_devfill
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 */
function rk_altfill(buffer, size, strong, state) {
    var err;

    err = rk_devfill(buffer, size, strong);
    if (err) {
        rk_fill(buffer, size, state);
    }
    return err;
}

/*
 * return a random gaussian deviate with variance unity and zero mean.
 */
function rk_gauss(state) {
    if (state.has_gauss) {
        var tmp = state.gauss;
        state.gauss = 0;
        state.has_gauss = 0;
        return tmp;
    } else {
        var f, x1, x2, r2;

        do {
            x1 = 2.0 * rk_double(state) - 1.0;
            x2 = 2.0 * rk_double(state) - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 === 0.0);

        /* Box-Muller transform */
        f = Math.sqrt(-2.0 * Math.log(r2) / r2);
        /* Keep for next call */
        state.gauss = f * x1;
        state.has_gauss = 1;
        return f * x2;
    }
}

/* ******************************************************************/
/*                                                                  */
/*                          iniarray.c                              */
/*                                                                  */
/* ******************************************************************/
/* initializes mt[RK_STATE_LEN] with a seed */
function init_genrand(self, s) {
    var mti;
    var mt = self.key;

    mt[0] = s & 0xffffffff;
    for (mti = 1; mti < RK_STATE_LEN; mti++) {
        /*
         * See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
         * In the previous versions, MSBs of the seed affect
         * only MSBs of the array mt[].
         * 2002/01/09 modified by Makoto Matsumoto
         */
        mt[mti] = (1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >>> 30)) + mti);
        /* for > 32 bit machines */
        mt[mti] &= 0xffffffff;
    }
    self.pos = mti;
    return;
}

/*
 * initialize by an array with array-length
 * init_key is the array for initializing keys
 * key_length is its length
 */
function init_by_array(self, init_key, key_length) {
    // init_by_array(rk_state *self, unsigned long init_key[], npy_intp key_length)
    /* was signed in the original code. RDH 12/16/2002 */
    var i = 1; // npy_intp
    var j = 0; // npy_intp
    var mt = self.key;
    var k; // npy_intp

    init_genrand(self, 19650218);
    k = (RK_STATE_LEN > key_length ? RK_STATE_LEN : key_length);
    for (; k; k--) {
        /* non linear */
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >>> 30)) * 1664525))
            + init_key[j] + j;
        /* for > 32 bit machines */
        mt[i] &= 0xffffffff;
        i++;
        j++;
        if (i >= RK_STATE_LEN) {
            mt[0] = mt[RK_STATE_LEN - 1];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = RK_STATE_LEN - 1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >>> 30)) * 1566083941))
             - i; /* non linear */
        mt[i] &= 0xffffffff; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= RK_STATE_LEN) {
            mt[0] = mt[RK_STATE_LEN - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000; /* MSB is 1; assuring non-zero initial array */
    self.gauss = 0;
    self.has_gauss = 0;
    self.has_binomial = 0;
}

/* eslint-disable camelcase, comma-dangle, no-underscore-dangle, strict */
/* ****************************************************************************/
/* Base numpy.random module entry,                                            */
/*                                                                            */
/* requires functions from randomkit.js and initarray.js                      */
/* ****************************************************************************/

var rk_state = {
    key: [],
    pos: null,
    has_gauss: null,
    gauss: null,
};

// imports
var np = Sk.importModule('numpy');

/**
 *  Class Name Identifier for RandomState
 */
var CLASS_RANDOMSTATE = 'RandomState';

function cont0_array(state, func, size, lock) {
    // not implemented
    var array_data;
    var array;
    var length;
    var i;

    // we just return a single value!
    if (Sk.builtin.checkNone(size)) {
        return new Sk.builtin.float_(func.call(null, state));
    }

    array = Sk.misceval.callsim(np.$d.empty, size, Sk.builtin.float_);
    length = array.v.buffer.length;

    array_data = array.v.buffer; // data view on the ndarray

    for (i = 0; i < length; i++) {
        array_data[i] = func.call(null, state);
    }

    return array;
}

function cont1_array_sc(state, func, size, a, lock) {
    // not implemented
}

/**
 *  This is the actual module nump.random
 */
var $builtinmodule = function(name) {
    var mod = {};

    var randomState_c = function($gbl, $loc) {
        var js__init__ = function(self, seed) {
            debugger;
            if (seed == null) {
                seed = Sk.builtin.none.none$;
            }

            self.internal_state = createRkState();

            // poisson_lam_max = np.iinfo('l').max - np.sqrt(np.iinfo('l').max)*10
            self.poisson_lam_max = new Sk.builtin.int_(Math.pow(2, 53) - 1);

            self.lock = null; // Todo self.lock = Lock()
            Sk.misceval.callsim(self.seed, self, seed); // self.seed(seed);
        };
        js__init__.co_varnames = ['self', 'seed'];
        js__init__.$defaults = [Sk.builtin.none.none$];
        $loc.__init__ = new Sk.builtin.func(js__init__);
        /*
            seed(seed=None)
            Seed the generator.
            This method is called when `RandomState` is initialized. It can be
            called again to re-seed the generator. For details, see `RandomState`.
            Parameters
            ----------
            seed : int or array_like, optional
                Seed for `RandomState`.
                Must be convertible to 32 bit unsigned integers.
            See Also
            --------
            RandomState
        */
        $loc.seed = new Sk.builtin.func(function(self, seed) {
            if (seed == null) {
                seed = Sk.builtin.none.none$;
            }

            var errcode; // rk_error
            var obj; // ndarray

            try {
                if (Sk.builtin.checkNone(seed)) {
                    // with self.lock
                    errcode = rk_randomseed(self.internal_state);
                } else {
                    var idx = new Sk.builtin.int_(Sk.misceval.asIndex(seed)); // ToDo: operator.index(seed)
                    var js_idx = Sk.ffi.remapToJs(idx);

                    if (js_idx > Math.pow(2, 32) - 1 || js_idx < 0) {
                        throw new Sk.builtin.ValueError('Seed must be between 0 and 4294967295');
                    }

                    // with self.lock
                    rk_seed(js_idx, self.internal_state);
                }
            } catch(e) {
                if (e instanceof Sk.builtin.TypeError) {
                    // ToDo: pass in dtype to asarray
                    obj = Sk.misceval.callsim(np.$d.asarray, seed, Sk.builtin.int_);

                    /*
                    if ((obj > Math.pow(2, 32) - 1) | (obj < 0).any()) {
                        throw new Sk.builtin.ValueError('Seed must be between 0 and 4294967295');
                    }
                    */

                    // check for each item in array
                    obj.v.buffer.map(function(elem) {
                        if (elem > Math.pow(2, 32) - 1 || elem < 0) {
                            throw new Sk.builtin.ValueError('Seed must be between 0 and 4294967295');
                        }
                    });

                    // our numpy module does not support astype
                    // obj = obj.astype('L', casting='unsafe')
                    // with self.lock:

                    // init_by_array(self.internal_state, <unsigned long *>PyArray_DATA(obj), PyArray_DIM(obj, 0))
                    // last parameter is the key_length of first dim!
                    init_by_array(self.internal_state, obj.v.buffer, obj.v.shape[0]);
                } else {
                    throw e;
                }
            }
        });

        $loc.set_state = new Sk.builtin.func(function(self) {

        });

        $loc.get_state = new Sk.builtin.func(function(self) {
            // save current state as ndarray
            // remap internal_state.key to Python objects and then call
            var js_key = self.internal_state.key.map(function(elem) {
                return new Sk.builtin.int_(elem);
            });

            var state = obj = Sk.misceval.callsim(np.$d.asarray, new Sk.builtin.tuple(js_key), Sk.builtin.int_);

            var has_gauss = new Sk.builtin.int_(self.internal_state.has_gauss);
            var gauss  = new Sk.builtin.float_(self.internal_state.gauss );
            var pos  = new Sk.builtin.int_(self.internal_state.pos);

            return new Sk.builtin.tuple([new Sk.builtin.str('MT19937'), state, pos, has_gauss, gauss]);
        });

        /*
        random_sample(size=None)
        Return random floats in the half-open interval [0.0, 1.0).
        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::
          (b - a) * random_sample() + a
        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).
        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098
        >>> type(np.random.random_sample())
        <type 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])
        Three-by-two array of random numbers from [-5, 0):
        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])
        */
        var js_random_sample = function(self, size) {
            // get *args
            if (size == null) {
                size = Sk.builtin.none.none$;
            }

            var py_res = cont0_array(self.internal_state, rk_double, size, self.lock);

            return py_res;
        };
        js_random_sample.co_varnames = ['self', 'size'];
        js_random_sample.$defaults = [Sk.builtin.none.none$];
        $loc.random_sample = new Sk.builtin.func(js_random_sample);

        var js_tomaxint = function(self, size) {
            throw new NotImplementedError('RandomState.tomaxint');
            // return disc0_array(self.internal_state, rk_long, size, self.lock)
        };
        js_tomaxint.co_varnames = ['self', 'size'];
        js_tomaxint.$defaults = [Sk.builtin.none.none$];
        $loc.tomaxint = new Sk.builtin.func(js_tomaxint);

        /*
        randint(low, high=None, size=None)
        Return random integers from `low` (inclusive) to `high` (exclusive).
        Return random integers from the "discrete uniform" distribution in the
        "half-open" interval [`low`, `high`). If `high` is None (the default),
        then results are from [0, `low`).
        */
        var js_randint = function(self, low, high, size) {
            throw new NotImplementedError('RandomState.randint');
        };
        js_randint.co_varnames = ['self', 'low', 'high', 'size'];
        js_randint.$defaults = [null, Sk.builtin.none.none$, Sk.builtin.none.none$];
        $loc.randint = new Sk.builtin.func(js_randint);

        $loc.rand = new Sk.builtin.func(function(self) {
            // get *args
            args = new Sk.builtins.tuple(Array.prototype.slice.call(arguments, 1));
            if (args.v.length === 0) {
                return Sk.misceval.callsim(self.random_sample, self);
            }

            return Sk.misceval.callsim(self.random_sample, self, args);
        });

        var js_bytes = function(self, length) {
            throw new NotImplementedError('RandomState.bytes');
        };
        $loc.bytes = new Sk.builtin.func(js_bytes);

        var js_choice = function(self, length) {
            throw new NotImplementedError('RandomState.choice');
        };
        $loc.choice = new Sk.builtin.func(js_choice);

        var js_uniform = function(self, length) {
            throw new NotImplementedError('RandomState.uniform');
        };
        $loc.uniform = new Sk.builtin.func(js_uniform);

        $loc.randn = new Sk.builtin.func(function(self) {
            args = new Sk.builtins.tuple(Array.prototype.slice.call(arguments, 1));
            if (args.v.length === 0) {
                return Sk.misceval.callsim(self.standard_normal, self);
            }

            return Sk.misceval.callsim(self.standard_normal, self, args);
        });

        $loc.tp$getattr = Sk.builtin.object.prototype.GenericGetAttr;

        $loc.tp$setattr = Sk.builtin.object.prototype.GenericSetAttr;
    };


    mod[CLASS_RANDOMSTATE] = Sk.misceval.buildClass(mod, randomState_c,
      CLASS_RANDOMSTATE, []);


    // _rand is just an instance of the RandomState class!
    mod._rand = Sk.misceval.callsim(mod[CLASS_RANDOMSTATE]);

    // map _rand.rand
    mod.rand = Sk.abstr.gattr(mod._rand, 'rand', true);
    mod.seed = Sk.abstr.gattr(mod._rand, 'seed', true);
    mod.random_sample = Sk.abstr.gattr(mod._rand, 'random_sample', true);
    mod.random = mod.random_sample;
    mod.sample = mod.random_sample;


    return mod;
};
