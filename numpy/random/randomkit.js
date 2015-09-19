/* eslint-disable camelcase, comma-dangle, no-underscore-dangle, strict */

var RK_STATE_LEN = 624;

var rk_state_ = {
    key: [],
    pos: null,
    has_gauss: null, /* !=0 gauss contains a gaussion deviate */

    /* The rk_state structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */

    has_binominal: null, /* !=0 following parameters initialized for binomial */

    psave: null,
    nsave: null,
    r: null,
    q: null,
    fm: null,
    m: null,
    p1: null,
    xm: null,
    xl: null,
    xr: null,
    laml: null,
    lamr: null,
    p2: null,
    p3: null,
    p4: null,
};

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

/* Thomas Wang 32 bits integer hash function */
function rk_hash(key) {
    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);
    return key;
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
        seed = (1812433253 * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffff;
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

    if (rk_devfill(state.key, null, 0) == rk_error.RK_NOERR) {
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

    // We cannot access the clock in javascript
    // We try to use the high performance timer, that is only supported in modern
    // browser versions -> No IE9 support
    if (window.performance && typeof window.performance.now === 'function') {
        var clock_stub = window.performance.now();
        rk_seed(rk_hash(tv.getTime()) ^ rk_hash(tv.getMilliseconds()) ^ rk_hash(clock_stub));
    } else {
        rk_seed(rk_hash(tv.getTime()) ^ rk_hash(tv.getMilliseconds()));
    }

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

        for(i = 0; i < N - M; i++) {
            y = (state.key[i] & UPPER_MASK) | (state.key[i + 1] & LOWER_MASK);
            state.key[i] = state.key[i + M] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
        }

        for (; i < N - 1; i++) {
            y = (state.key[i] & UPPER_MASK) | (state.key[i + 1] & LOWER_MASK);
            state.key[i] = state.key[i + (M - N)] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
        }
        y = (state.key[N - 1] & UPPER_MASK) | (state.key[0] & LOWER_MASK);
        state.key[N - 1] = state.key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

        state.pos = 0;
    }

    y = state.key[state.pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}

/*
 * Returns a random long between 0 and LONG_MAX inclusive
 */
function rk_long(state) {
    return rk_ulong(state) >> 1;
}

/*
 * Returns a random unsigned long between 0 and ULONG_MAX inclusive
 */
function rk_ulong(state) {
    return (rk_random(state) << 32) | (rk_random(state));
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
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;

    if (max <= 0xffffffff) {
        while ((value = (rk_random(state) & mask)) > max) {
            // empty block
        }
    }
    else {
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
    var a = rk_random(state) >> 5, b = rk_random(state) >> 6;
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
        buf[i++] = (r >> 8) & 0xFF;
        buf[i++] = (r >> 16) & 0xFF;
        buf[i++] = (r >> 24) & 0xFF;
    }

    if (!size) {
        return;
    }

    r = rk_random(state);
    for (; size; r >>= 8, size--) {
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
    return RK_ENODEV;
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
