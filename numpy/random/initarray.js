/* eslint-disable camelcase, comma-dangle, no-underscore-dangle, strict */
/**
 * Functinos from initarray.c and initarray.h
 */

// TODO: import numpy array here!
// ToDO: maybe change self.* to self.v.*

var RK_STATE_LEN = 624;

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
        mt[mti] = (1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
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
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525))
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
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941))
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
