/* eslint-disable camelcase, no-eq-null, comma-dangle, no-underscore-dangle, strict, new-cap, no-var, vars-on-top, no-param-reassign, func-names */
/**
 * This may be removed after the PR #518 has been merged
 */
Sk.misceval.tryCatch = Sk.misceval.tryCatch || function(tryFn, catchFn) {
    var r;

    try {
        r = tryFn();
    } catch(e) {
        return catchFn(e);
    }

    if (r instanceof Sk.misceval.Suspension) {
        var susp = new Sk.misceval.Suspension(undefined, r);
        susp.resume = function() { return Sk.misceval.tryCatch(r.resume, catchFn); };
        return susp;
    } else {
        return r;
    }
};

/* ****************************************************************************/
/*                                  RandomKit                                 */
/*                                                                            */
/* ****************************************************************************/
var RK_STATE_LEN = 624;

function createRkState(key, pos, has_gauss, has_binomial, psave, nsave, r, q, fm, m, p1, xm, xl, xr, laml, lamr, p2, p3, p4) {
    return {
        key: key || new Array(RK_STATE_LEN),
        pos: pos || null,
        gauss: null,
        has_gauss: has_gauss || null, /* !=0 gauss contains a gaussion deviate */

        /* The rk_state structure has been extended to store the following
         * information for the binomial generator. If the input values of n or p
         * are different than nsave and psave, then the other parameters will be
         * recomputed. RTK 2005-09-02 */

        has_binomial: has_binomial || null, /* !=0 following parameters initialized for binomial */

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

var RK_MAX = 4294967296.0; // old value0xFFFFFFFF;

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
    var s;

    state.key[0] = seed >>> 0;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 1; pos < RK_STATE_LEN; pos++) {
        s = state.key[pos - 1] ^ (state.key[pos - 1] >>> 30);
        state.key[pos] = (((((s & 0xffff0000) >>> 16) * 1812433253) << 16) + (s & 0x0000ffff) * 1812433253) + pos;
        state.key[pos] >>>= 0;
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
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
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

function rk_binomial_btpe(state, n, p) {
    var r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
    var a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
    var m, y, k, i;

    if (!(state.has_binomial === 1) || (state.nsave != n) || (state.psave != p)) {
        // initialize
        state.nsave = n;
        state.psave = p;
        state.has_binomial = 1;
        state.r = r = Math.min(p, 1.0 - p);
        state.q = q = 1.0 - r;
        state.fm = fm = n * r + r;
        state.m = m = Math.floor(state.fm);
        state.p1 = p1 = Math.floor(2.195 * Math.sqrt(n * r * q) - 4.6 * q) + 0.5;
        state.xm = xm = m + 0.5;
        state.xl = xl = xm - p1;
        state.xr = xr = xm + p1;
        state.c = c = 0.134 + 20.5 / (15.3 + m);
        a = (fm - xl) / (fm - xl * r);
        state.laml = laml = a * (1.0 + a / 2.0);
        a = (xr - fm) / (xr * q);
        state.lamr = lamr = a * (1.0 + a / 2.0);
        state.p2 = p2 = p1 * (1.0 + 2.0 * c);
        state.p3 = p3 = p2 + c / laml;
        state.p4 = p4 = p3 + c / lamr;
    } else {
        r = state.r;
        q = state.q;
        fm = state.fm;
        m = state.m;
        p1 = state.p1;
        xm = state.xm;
        xl = state.xl;
        xr = state.xr;
        c = state.c;
        laml = state.laml;
        lamr = state.lamr;
        p2 = state.p2;
        p3 = state.p3;
        p4 = state.p4;
    }

    // use while loop with case statement
    var goto_label = 'Step10';
    goto_loop: while (goto_label) {
        switch (goto_label) {
        case 'Step10':
            nrq = n * r * q;
            u = rk_double(state) * p4;
            v = rk_double(state);
            if (u > p1) {
                goto_label = 'Step20';
                continue goto_loop;
            }
            y = Math.floor(xm - p1 * v + u);
            goto_label = 'Step60';
            continue goto_loop;
        case 'Step20':
            if (u > p2) {
                goto_label = 'Step30';
                continue goto_loop;
            }

            x = xl + (u - p1) / c;
            v = v * c + 1.0 - Math.abs(m - x + 0.5) / p1;
            if (v > 1.0) {
                goto_label = 'Step10';
                continue goto_loop;
            }
            y = Math.floor(x);
            goto_label = 'Step50';
            continue goto_loop;
        case 'Step30':
            if (u > p3) {
                goto_label = 'Step40';
                continue goto_loop;
            }
            y = Math.floor(xl + Math.log(v) / laml);
            if (y < 0) {
                goto_label = 'Step10';
                continue goto_loop;
            }
            v = v * (u - p2) * laml;
            goto_label = 'Step50';
            continue goto_loop;
        case 'Step40':
            y = Math.floor(xr - Math.log(v) / lamr);
            if (y > n) {
                goto_label = 'Step10';
                continue goto_loop;
            }
            v = v * (u - p3) * lamr;
            // case fallthrough as in L335
        case 'Step50':
            k = Math.abs(y - m);
            if ((k > 20) && (k < ((nrq) / 2.0 - 1))) {
                goto_label = 'Step52';
                continue goto_loop;
            }
            s = r / q;
            a = s * (n + 1);
            F = 1.0;
            if (m < y) {
                for (i = m + 1; i <= y; i++) {
                    F *= (a / i - s);
                }
            } else if (m > y) {
                for (i = y + 1; i <= m; i++) {
                    F /= (a / i - s);
                }
            }
            if (v > F) {
                goto_label = 'Step10';
                continue goto_loop;
            }
            goto_label = 'Step60';
            continue goto_loop;
        case 'Step52':
            rho = (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
            t = -k * k / (2 * nrq);
            A = Math.log(v);
            if (A < (t - rho)) {
                goto_label = 'Step60';
                continue goto_loop;
            }
            if (A > (t + rho)) {
                goto_label = 'Step10';
                continue goto_loop;
            }

            x1 = y + 1;
            f1 = m + 1;
            z = n + 1 - m;
            w = n - y + 1;
            x2 = x1 * x1;
            f2 = f1 * f1;
            z2 = z * z;
            w2 = w * w;
            if (A > (xm * Math.log(f1 / x1)
                   + (n - m + 0.5) * Math.log(z / w)
                   + (y - m) * Math.log(w * r / (x1 * q))
                   + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
                   + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
                   + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
                   + (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0)) {
                goto_label = 'Step10';
                continue goto_loop;
            }
            // case fallthrough like in L386
        case 'Step60':
            if (p > 0.5) {
                y = n - y;
            }
            goto_label = false;
            break;
        default:
            console.log("unhandeled case: " + goto_label);
            break;
        }
    }

    return y;
}

function rk_binomial_inversion(state, n, p) {
    var q;
    var qn;
    var np;
    var px;
    var U;
    var X;
    var bound;

    if (!(state.has_binomial === 1) ||
         (state.nsave != n) ||
         (state.psave != p)) {
        state.save = n;
        state.psave = p;
        state.has_binomial = 1;
        state.q = q = 1.0 - p;
        state.r = qn = Math.exp(n * Math.log(q));
        state.c = np = n * p;
        state.m = bound = Math.min(n, np + 10.0 * Math.sqrt(np * q + 1));
    } else {
        q = state.q;
        qn = state.r;
        np = state.c;
        bound = state.m;
    }

    X = 0;
    px = qn;
    U = rk_double(state);
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = rk_double(state);
        } else {
            U -= px;
            px  = ((n - X + 1) * p * px) / (X * q);
        }
    }

    return X;
}

function rk_binomial(state, n, p) {
    var q;
    if (p <= 0.5) {
        if (p * n <= 30.0) {
            return rk_binomial_inversion(state, n, p);
        } else {
            return rk_binomial_btpe(state, n, p);
        }
    } else {
        q = 1.0 - p;
        if (q * n <= 30.0) {
            return n - rk_binomial_inversion(state, n, q);
        } else {
            return n - rk_binomial_btpe(state, n, q);
        }
    }
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
        array_data[i] = new Sk.builtin.float_(func.call(null, state));
    }

    return array;
}

function cont1_array_sc(state, func, size, a, lock) {
    // not implemented
}

// TODO:
function discnp_array_sc(state, func, size, n, p, lock) {
    var array_data = [];
    var array;
    var length;
    var i;
    var jsn = Sk.ffi.remapToJs(n);
    var jsp = Sk.ffi.remapToJs(p);

    if (Sk.builtin.checkNone(size)) {
        return new Sk.builtin.int_(func(state, jsn, jsp));
    } else {
        array = Sk.misceval.callsim(np.$d.empty, size, Sk.builtin.int_);
        length = Sk.builtin.len(array).v;
        array_data = array.v.buffer;

        for (i = 0; i < length; i++) {
            array_data[i] = new Sk.builtin.int_(func(state, jsn, jsp));
        }

        return array;
    }
}

// https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx#L356
function discnp_array(state, func, size, on, op, lock) {
    var array_data = [];
    var array;
    var length;
    var i;
    var op_data;
    var on_data;
    var multi; // broadcast
    var on1;
    var op1;

    if (Sk.builtin.checkNone(size)) {
        multi = null; // ToDo: MultiIter is not supported
        var py_shape = new Sk.builtin.tuple(on.v.shape.map(
            function(x) {
                return new Sk.builtin.int_(x);
            }));

        array = Sk.misceval.callsim(np.$d.empty, py_shape, Sk.builtin.int_);
        array_data = array.v.buffer; // ToDo: use PyArray_DATA

        // broadcast arrays
        on1 = on.v.buffer;
        op1 = op.v.buffer;

        if (op1.length !== on1.length) {
            if (op1.length === 1) {
                for (i = 1; i < on1.length; i++) {
                    op1.push(op1[0]);
                }
            } else if (on1.length === 1) {
                for (i = 1; i < op1.length; i++) {
                    on1.push(on1[0]);
                }                
            } else {
                throw new Sk.builtin.ValueError("cannot broadcast n and p to a common shape");
            }
        }

        for (i = 0; i < array_data.length; i++) {
            on_data = Sk.ffi.remapToJs(on1[i]);
            op_data = Sk.ffi.remapToJs(op1[i]);
            array_data[i] = new Sk.builtin.int_(func(state, on_data, op_data));
        }
    } else {
        array = Sk.misceval.callsim(np.$d.empty, size, Sk.builtin.int_);
        array_data = array.v.buffer; // PyArray_DATA() TODO
        // multi = PyArray_MultiIterNew(3, array, on, op);
        on1 = on.v.buffer;
        op1 = op.v.buffer;

        // ToDo: use PyArray_SIZE
        if (array_data.length !== on1.length && array_data.length !== op1.length) {
            throw new Sk.builtin.ValueError("size is not compatible with inputs");
        }
        // this loop assumes the same shape and array order
        for (i = 0; i < array_data.length; i++) {
            on_data = Sk.ffi.remapToJs(on1[i]);
            op_data = Sk.ffi.remapToJs(op1[i]);
            array_data[i] = new Sk.builtin.int_(func(state, on_data, op_data));
        }
    }

    return array;
}

function PyArray_FROM_OTF(m, type, flags) {
    // ToDo: pass in the flags if available
    return Sk.misceval.callsim(np.$d.array, m, type);
}

/**
 *  This is the actual module nump.random
 */
var $builtinmodule = function(name) {
    var mod = {};

    var randomState_c = function($gbl, $loc) {
        var js__init__ = function(self, seed) {
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
            Sk.builtin.pyCheckArgs("randint", arguments, 1, 3, true);
            if (size == null) {
                size = Sk.builtin.none.none$;
            }

            var lo;
            var hi;
            var rv;
            var diff;
            var array_data = [];
            var array;
            var length;
            var i;

            if (high == null || Sk.builtin.checkNone(high)) {
                lo = new Sk.builtin.int_(0);
                hi = new Sk.builtin.int_(low);
            } else {
                lo = new Sk.builtin.int_(low);
                hi = new Sk.builtin.int_(high);
            }

            lo = Sk.ffi.remapToJs(lo);
            hi = Sk.ffi.remapToJs(hi);

            if (lo >= hi) {
                throw new Sk.builtin.ValueError("low >= high");
            }

            diff = Math.abs(hi - lo - 1);

            if (Sk.builtin.checkNone(size)) {
                rv = new Sk.builtin.int_(lo + rk_interval(diff, self.internal_state));
                return rv;
            } else {
                array = Sk.misceval.callsim(np.$d.empty, size, Sk.builtin.int_);
                length = Sk.builtin.len(array).v; // get size
                array_data = array.v.buffer;

                for (i = 0; i < length; i++) {
                    rv = lo + rk_interval(diff, self.internal_state);
                    // ToDo: do we need some casting here?
                    array_data[i] = new Sk.builtin.int_(rv);
                }

                return array;
            }
        };
        js_randint.co_varnames = ['self', 'low', 'high', 'size'];
        js_randint.$defaults = [Sk.builtin.none.none$, Sk.builtin.none.none$, Sk.builtin.none.none$];
        $loc.randint = new Sk.builtin.func(js_randint);

        var js_random_integers = function(self, low, high, size) {
            if (high == null || Sk.builtin.checkNone(high)) {
                high = low;
                low = new Sk.builtin.int_(1);
            }

            return Sk.misceval.callsim(self.randint, self, low, sum = Sk.abstr.numberBinOp(high, new Sk.builtin.int_(1), 'Add'), size);
        };
        js_random_integers.co_varnames = ['self', 'low', 'high', 'size'];
        js_random_integers.$defaults = [Sk.builtin.none.none$, Sk.builtin.none.none$, Sk.builtin.none.none$];
        $loc.random_integers = new Sk.builtin.func(js_random_integers);

        $loc.rand = new Sk.builtin.func(function(self) {
            // get *args
            args = new Sk.builtins.tuple(Array.prototype.slice.call(arguments, 1));
            if (args.v.length === 0) {
                return Sk.misceval.callsim(self.random_sample, self);
            }

            return Sk.misceval.callsim(self.random_sample, self, args);
        });

        // binomial: mtrand.pyx:L3587
        var js_binomial = function(self, n, p, size) {
            Sk.builtin.pyCheckArgs("binomial", arguments, 2, 3, true);
            var on; // ndarray
            var op; // ndarray
            var ln; // long
            var fp; // double

            if (size == null) {
                size = Sk.builtin.none.none$;
            }
            debugger;
            var ex = null;
            try {
                fp = Sk.ffi.remapToJs(new Sk.builtin.float_(p));
                ln = Sk.ffi.remapToJs(new Sk.builtin.int_(n));
            } catch(e) {
                ex = e;
            }

            // check if the conversion was successful
            if (ex === null) {
                if (ln < 0) {
                    throw new Sk.builtin.ValueError("n < 0");
                }
                if (fp < 0) {
                    throw new Sk.builtin.ValueError("p < 0");
                } else if (fp > 1) {
                    throw new Sk.builtin.ValueError("p > 1");
                } else if (isNaN(fp)) {
                    throw new Sk.builtin.ValueError("p is nan");
                }
                return discnp_array_sc(self.internal_state, rk_binomial, size, ln, fp, self.lock);
            }

            // we may have to deal with arrays
            on = PyArray_FROM_OTF(n, Sk.builtin.int_);
            op = PyArray_FROM_OTF(p, Sk.builtin.float_);
            var py_zero = new Sk.builtin.int_(0);
            if (Sk.misceval.callsim(np.$d.any, Sk.misceval.callsim(np.$d.less, n, py_zero)) == Sk.builtin.bool.true$) {
                throw new Sk.builtin.ValueError("n < 0");
            }
            if (Sk.misceval.callsim(np.$d.any, Sk.misceval.callsim(np.$d.less, p, py_zero)) == Sk.builtin.bool.true$) {
                throw new Sk.builtin.ValueError("p < 0");
            }
            if (Sk.misceval.callsim(np.$d.any, Sk.misceval.callsim(np.$d.greater, p, new Sk.builtin.int_(1))) == Sk.builtin.bool.true$) {
                throw new Sk.builtin.ValueError("p > 1");
            }

            // ToDo: we need to broadcast the array
            return discnp_array(self.internal_state, rk_binomial, size, on, op, self.lock);
        };
        $loc.binomial = new Sk.builtin.func(js_binomial);

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
    mod.binomial = Sk.abstr.gattr(mod._rand, 'binomial', true);
    mod.randint = Sk.abstr.gattr(mod._rand, 'randint', true);
    mod.random_integers = Sk.abstr.gattr(mod._rand, 'random_integers', true);


    return mod;
};
