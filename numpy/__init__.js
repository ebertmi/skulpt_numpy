/**
 * ES6 - Math polyfill, when .dot is implemented, we do not need to rely on mathjs anymore
 * borrowed from: https://github.com/MaxArt2501/es6-math
 */
(function (factory) {
    if (typeof define === "function" && define.amd) {
        // AMD. Register as an anonymous module.
        define([], factory);
    } else factory();
})(function() {
    "use strict";
    // x | 0 is the simplest way to implement ToUint32(x)
    var M = Math,
        N = Number,
        prop, def = Object.defineProperty,
        mathXtra = {
            // Hyperbolic functions
            sinh: function sinh(x) {
                // If -0, must return -0.
                if (x === 0) return x;
                var exp = M.exp(x);
                return exp/2 - .5/exp;
            },
            cosh: function cosh(x) {
                var exp = M.exp(x);
                return exp/2 + .5/exp;
            },
            tanh: function tanh(x) {
                // If -0, must return -0.
                if (x === 0) return x;
                // Mathematically speaking, the formulae are equivalent.
                // But computationally, it's better to make exp tend to 0
                // rather than +Infinity
                if (x < 0) {
                    var exp = M.exp(2 * x);
                    return (exp - 1) / (exp + 1);
                } else {
                    var exp = M.exp(-2 * x);
                    return (1 - exp) / (1 + exp);
                }
            },
            asinh: function asinh(x) {
                return x === -Infinity ? -Infinity : M.log(x + M.sqrt(x * x + 1));
            },
            acosh: function acosh(x) {
                return x >= 1 ? M.log(x + M.sqrt(x * x - 1)) : NaN;
            },
            atanh: function atanh(x) {
                return x >= -1 && x <= 1 ? M.log((1 + x) / (1 - x)) / 2 : NaN;
            },

            // Exponentials and logarithms
            expm1: function expm1(x) {
                // If -0, must return -0. But Math.exp(-0) - 1 yields +0.
                return x === 0 ? x : M.exp(x) - 1;
            },
            log10: function log10(x) {
                return M.log(x) / M.LN10;
            },
            log2: function log2(x) {
                return M.log(x) / M.LN2;
            },
            log1p: function log1p(x) {
                // If -0, must return -0. But Math.log(1 + -0) yields +0.
                return x === 0 ? x : M.log(1 + x);
            },

            // Various
            sign: function sign(x) {
                // If -0, must return -0.
                return isNaN(x) ? NaN : x < 0 ? -1 : x > 0 ? 1 : +x;
            },
            cbrt: function cbrt(x) {
                // If -0, must return -0.
                return x === 0 ? x : x < 0 ? -M.pow(-x, 1/3) : M.pow(x, 1/3);
            },
            hypot: function hypot(value1, value2) { // Must have a length of 2
                for (var i = 0, s = 0, args = arguments; i < args.length; i++)
                    s += args[i] * args[i];
                return M.sqrt(s);
            },

            // Rounding and 32-bit operations
            trunc: function trunc(x) {
                return x === 0 ? x : x < 0 ? M.ceil(x) : M.floor(x);
            },
            fround: typeof Float32Array === "function"
                    ? (function(arr) {
                        return function fround(x) { return arr[0] = x, arr[0]; };
                    })(new Float32Array(1))
                    : function fround(x) { return x; },

            clz32: function clz32(x) {
                if (x === -Infinity) return 32;
                if (x < 0 || (x |= 0) < 0) return 0;
                if (!x) return 32;
                var i = 31;
                while (x >>= 1) i--;
                return i;
            },
            imul: function imul(x, y) {
                return (x | 0) * (y | 0) | 0;
            }
        },
        numXtra = {
            isNaN: function isNaN(x) {
                // NaN is the only Javascript object such that x !== x
                // The control on the type is for eventual host objects
                return typeof x === "number" && x !== x;
            },
            isFinite: function isFinite(x) {
                return typeof x === "number" && x === x && x !== Infinity && x !== -Infinity;
            },
            isInteger: function isInteger(x) {
                return typeof x === "number" && x !== Infinity && x !== -Infinity && M.floor(x) === x;
            },
            isSafeInteger: function isSafeInteger(x) {
                return typeof x === "number" && x > -9007199254740992 && x < 9007199254740992 && M.floor(x) === x;
            },
            parseFloat: parseFloat,
            parseInt: parseInt
        },
        numConsts = {
            EPSILON: 2.2204460492503130808472633361816e-16,
            MAX_SAFE_INTEGER: 9007199254740991,
            MIN_SAFE_INTEGER: -9007199254740991
        };

    for (prop in mathXtra)
        if (typeof M[prop] !== "function")
            M[prop] = mathXtra[prop];

    for (prop in numXtra)
        if (typeof N[prop] !== "function")
            N[prop] = numXtra[prop];

    try {
        prop = {};
        def(prop, 0, {});
        for (prop in numConsts)
            if (!(prop in N))
                def(N, prop, {value: numConsts[prop]});
    } catch (e) {
        for (prop in numConsts)
            if (!(prop in N))
                N[prop] = numConsts[prop];
    }
});
var $builtinmodule = function (name) {
    /**
        Made by Michael Ebert for https://github.com/skulpt/skulpt
        ndarray implementation inspired by https://github.com/geometryzen/davinci-dev (not compatible with skulpt)

        Some methods are based on the original numpy implementation.

        See http://waywaaard.github.io/skulpt/ for more information.
    **/
    /* eslint-disable */

    var NPY_MAXDIMS = 32;

    var numpy = function () {
      if (typeof mathjs == 'function') {
        // load mathjs instance
        this.math = mathjs();
      } else {
        Sk.debugout('mathjs not included and callable');
      }
    };

    numpy.prototype.arange = function (start, stop, step) {
      if (step === undefined)
        step = 1.0;

      start *= 1.0;
      stop *= 1.0;
      step *= 1.0;

      var res = [];
      for (var i = start; i < stop; i += step) {
        res.push(i);
      }

      return res;
    };

    /*
     * This function checks to see if arr is a 0-dimensional array and, if so, returns the appropriate array scalar. It should be used whenever 0-dimensional arrays could be returned to Python.
     */
    function PyArray_Return(arr) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            var dim = PyArray_DIM(arr, 0);
            if (dim === 0) {
                return PyArray_DATA(arr)[0]; // return the only element
            } else {
                return arr;
            }
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    function PyArray_UNPACK_SHAPE(arr, shape) {
        var js_shape;
        if (Sk.builtin.checkInt(shape)) {
            js_shape = [Sk.ffi.remapToJs(shape)];
        } else {
            js_shape = Sk.ffi.remapToJs(shape);
        }

        // special auto shape infering
        if (js_shape[0] === -1) {
            if (js_shape.length === 1) {
                js_shape[0] = PyArray_SIZE(arr);
            } else {
                var n_size = prod(js_shape.slice(1));
                var n_firstDim = PyArray_SIZE(arr) / n_size;
                js_shape[0] = n_firstDim;
            }
        }

        if (prod(js_shape) !== PyArray_SIZE(arr)) {
            throw new ValueError('total size of new array must be unchanged');
        }

        return js_shape;
    }

    function PyArray_SIZE(arr) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            return prod(arr.v.shape);
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    function PyArray_DATA(arr) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            return arr.v.buffer;
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    function PyArray_STRIDES(arr) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            return computeStrides(arr.v.shape);
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    function PyArray_STRIDE(arr, n) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            var strides = computeStrides(arr.v.shape);
            return strides[n];
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    /*
     *  The number of dimensions in the array.
     *  Returns a javascript value
     */
    function PyArray_NDIM(arr) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            return arr.v.shape.length;
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    /*
     *  Returns a pointer to the dimensions/shape of the array. The number of elements matches the number of dimensions of the array.
     *  This returns javascript values!
     */
    function PyArray_DIMS(arr) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            return arr.v.shape;
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    /*
     *  Return the shape in the nth dimension.
     */
    function PyArray_DIM(arr, n) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            return arr.v.shape[n];
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    function PyArray_NewShape(arr, shape, order) {
        if (arr && (Sk.abstr.typeName(arr) === CLASS_NDARRAY)) {
            var py_shape = new Sk.builtin.tuple(shape.map(
              function (x) {
                return new Sk.builtin.int_(x);
            }));

             var py_order = Sk.ffi.remapToPy(order);
            return Sk.misceval.callsim(arr.reshape, arr, py_shape, py_order);
        } else {
            throw new Error('Internal API-Call Error occured in PyArray_NDIM.', arr);
        }
    }

    // vdot function for python basic numeric types
    // https://github.com/numpy/numpy/blob/467d4e16d77a2e7c131aac53c639e82b754578c7/numpy/core/src/multiarray/vdot.c
    /*
     *  ip1: vector 1
     *  ip2: vector 2
     *  is1: stride of vector 1
     *  is2: stride of vector 2
     *  op: new nd_array data buffer for the result
     *  n:  number of elements in ap1 first dim
     *  ignore: not used anymore, however still existing for old api calls
     *
     */
    function OBJECT_vdot(ip1, is1, ip2, is2, op, n, ignore){
        function tryConjugate(pyObj) {
            var f = pyObj.tp$getattr("conjugate");
            if (f) {
                return Sk.misceval.callsim(pyObj['conjugate'], pyObj);
            } else {
                return pyObj; // conjugate for non complex types is just the real part
            }
        }

        var i; // npy_intp
        var tmp0; // PyObject
        var tmp1; // PyObject
        var tmp2; // PyObject
        var tmp = null; // PyObject
        var tmp3; // PyObject **

        var ip1_i = 0;
        var ip2_i = 0;

        for (i = 0; i < n; i++, ip1_i += is1, ip2_i += is2) {
            if (ip1[ip1_i] == null || ip2[ip2_i] == null) {
                tmp1 = Sk.builtin.bool.false$;
            } else {
                // try to call the conjugate function / each numeric type can call this
                tmp0 = Sk.misceval.callsim(ip1[ip1_i]['conjugate'], ip1[ip1_i]);

                if (tmp0 == null) {
                    return;
                }

                tmp1 = Sk.abstr.numberBinOp(tmp0, ip2[ip2_i], 'Mult');
                if (tmp1 == null) {
                    return;
                }
            }

            if (i === 0) {
                tmp = tmp1;
            } else {
                tmp2 = Sk.abstr.numberBinOp(tmp, tmp1, 'Add');
                //tmp2 = tmp + tmp1; // PyNumber_Add

                if (tmp2 == null) {
                    return;
                }
                tmp = tmp2;
            }
        }

        tmp3 = op;
        tmp2 = tmp3;
        op[0] = tmp;
    }

    /**
     *  Basic dummy impl. The real numpy implementation is about 600 LoC and relies
     *  on the complete data type impl.
     *  We just do a basic checking
     *  obj: any object or nested sequence
     *  maxdims: maximum number of dimensions to check for dtype (recursive call)
     */
    function PyArray_DTypeFromObject(obj, maxdims) {
        // gets first element or null from a nested sequence
        function seqUnpacker(obj, mDims) {
            if (Sk.builtin.checkSequence(obj)) {
                var length = Sk.builtin.len(obj);
                if (Sk.builtin.asnum$(length) > 0) {
                    // ToDo check if element is also of type sequence
                    var element = obj.mp$subscript(0);

                    // if we have a nested sequence, we decrement the maxdims and call recursive
                    if (mDims > 0 && Sk.builtin.checkSequence(element)) {
                        return seqUnpacker(element, mDims -1);
                    } else {
                        return element;
                    }
                }
            } else {
                return obj;
            }
        }

        var dtype = null;
        if (obj === null) {
            throw new Error('Internal API-Call Error occured in PyArray_ObjectType');
        }

        if (maxdims == null) {
            maxdims = NPY_MAXDIMS;
        }

        // simple type
        if (Sk.builtin.checkNumber(obj)) {
            element = obj;
        } else if (Sk.builtin.checkSequence(obj)) {
            // sequence
            element = seqUnpacker(obj, maxdims);
        } else if (Sk.abstr.typeName(obj) === CLASS_NDARRAY) {
            // array
            var length = Sk.builtin.len(obj);
            if (Sk.builtin.asnum$(length) > 0) {
                var element = PyArray_DATA(obj)[0];
            } else {
                // get dtype from ndarray
                return obj.v.dtype;
            }
        }

        // ToDo: investigate if this throw may happen
        try {
            dtype = Sk.builtin.type(element);
        } catch (e) {
            // pass
        }

        return dtype;
    }

    var NPY_NOTYPE = null;
    var NPY_DEFAULT_TYPE = 2;

    // our basic internal hierarchy for types
    // we can only promote from lower to higher numbers
    var Internal_TypeTable = {
        'complex': 3,
        'complex_': 3,
        'complex64': 3,
        'float': 2,
        'float_': 2,
        'float64': 2,
        'int': 1,
        'int_': 1,
        'int64': 1,
        'bool': 0,
        'bool_': 0,
    };

    function Internal_DType_To_Num(dtype) {
        var name = Sk.abstr.typeName(dtype);
        var num = Internal_TypeTable[name];

        if (num == null) {
            return -1;
        }

        return num;
    }

    /**
     * Not the real impl. as we do not really implement numpys data types and
     * the 1000s LoC for that. We just use the basic python types.
     *
     * This function returns the 'constructor' for the given type number
     */
    function PyArray_DescrFromType(typenum) {
        switch(typenum) {
        case 3:
            return Sk.builtin.complex;
        case 2:
            return Sk.builtin.float_;
        case 1:
            return Sk.builtin.int_;
        case 0:
            return Sk.builtin.bool;
        default:
            return NPY_NOTYPE;
        }
    }

    /*
     *  This function is useful for determining a common type that two or more arrays can be converted to.
     *  It only works for non-flexible array types as no itemsize information is passed. The mintype argument
     *  represents the minimum type acceptable, and op represents the object that will be converted to an array.
     *  The return value is the enumerated typenumber that represents the data-type that op should have.
     */
    function PyArray_ObjectType(op, minimum_type) {
        // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_ResultType
        // this is only and approximate implementation and is not even close to
        // the real numpy internals, however totally sufficient for our needs

        var dtype;

        dtype = PyArray_DTypeFromObject(op, NPY_MAXDIMS);

        // maybe empty ndarray object?
        if (dtype != null) {
            var num = Internal_DType_To_Num(dtype);
            if (num >= minimum_type) {
                return num;
            } else if(num < minimum_type) {
                // can we convert one type into the other?
                if (num >= 0 && minimum_type <= 3) {
                    return minimum_type;
                } else {
                    return NPY_NOTYPE; // NPY_NOTYPE
                }
            }
        } else {
            return NPY_DEFAULT_TYPE;
        }
    }

    /*
     *  A synonym for PyArray_DIMS, named to be consistent with the ‘shape’ usage within Python.
     */
    function PyArrray_Shape(arr) {
        return PyArray_DIMS(arr);
    }

    /*
     *  Cast/Promote object to given dtype with some intelligent type handling
     */
    function PyArray_Cast_Obj(obj, dtype) {
        if (dtype instanceof Sk.builtin.none && Sk.builtin.checkNone(obj)) {
            return obj;
        } else {
            return Sk.misceval.callsim(dtype, obj);
        }
    }

    // PyObject* PyArray_FromAny(PyObject* op, PyArray_Descr* dtype, int min_depth, int max_depth, int requirements, PyObject* context)
    /*
     *  - op: is any value or sequence that will be converted to an array (Python Object)
     *  - dtype: is callable constructor for the type (ie Sk.builtin.int_) (Python Object)
     *  - min_depth: we may enfore a minimum of dimensions (Python Int)
     *  - max_depth: maximum of dimensions (Python Int)
     *  - requirements: same flags to set, we do not support those (int)
     *  - context: ???
     */
    function PyArray_FromAny(op, dtype, min_depth, max_depth, requirements, context) {
        if (op == null) {
            throw new Error('Internal PyArray_FromAny API-Call error. "op" must not be null.');
        }

        if (dtype == null || Sk.builtin.checkNone(dtype)) {
            dtype = PyArray_DTypeFromObject(op, NPY_MAXDIMS);
            if (dtype == null) {
                dtype = NPY_DEFAULT_TYPE;
            }
        }

        var elements = [];
        var state = {};
        state.level = 0;
        state.shape = [];

        if (Sk.abstr.typeName(op) == CLASS_NDARRAY) {
            elements = PyArray_DATA(op);
            state = {};
            state.level = 0;
            state.shape = PyArray_DIMS(op);;
        } else {
            // now get items from op and create a new buffer object.
            unpack(op, elements, state);
        }

        // apply dtype castings
        for (i = 0; i < elements.length; i++) {
            elements[i] = PyArray_Cast_Obj(elements[i], dtype);
        }

        // check for min_depth
        var ndmin = Sk.builtin.asnum$(min_depth);
        if (ndmin >= 0 && ndmin > state.shape.length) {
          var _ndmin_array = [];
          for (i = 0; i < ndmin - state.shape.length; i++) {
            _ndmin_array.push(1);
          }
          state.shape = _ndmin_array.concat(state.shape);
        }

        // call array method or create internal ndarray constructor
        var _shape = new Sk.builtin.tuple(state.shape.map(function (x) {
            return new Sk.builtin.int_(x);
        }));

        var _buffer = new Sk.builtin.list(elements);
        // create new ndarray instance
        return Sk.misceval.callsim(mod[CLASS_NDARRAY], _shape, dtype,
          _buffer);
    }

    /*NUMPY_API
     * Numeric.matrixproduct(a,v)
     * just like inner product but does the swapaxes stuff on the fly
     */
    // from multiarraymodule.c Line: 940
    function MatrixProdcut(op1, op2) {
        return this.MatrixProduct2(op1, op2, null);
    }

    /*NUMPY_API
     * Numeric.matrixproduct2(a,v,out)
     * just like inner product but does the swapaxes stuff on the fly
     * https://github.com/numpy/numpy/blob/d033b6e19fc95a1f1fd6592de8318178368011b1/numpy/core/src/multiarray/methods.c#L2037
     */
    function MatrixProdcut(op1, op2, out) {
        var ap1; // PyArrayObject
        var ap2; // PyArrayObject
        var ret = null; // PyArrayObject
        var i; // npy_intp (int pointer)
        var j; // npy_intp (int pointer)
        var l; // npy_intp (int pointer)
        var typenum; // int
        var nd; // int
        var axis; // int
        var matchDim; // int
        var is1; // npy_intp
        var is2; // npy_intp
        var os; // npy_intp
        var op; // char *op
        var dimensions = new Array(); // npy_intp dimensions[NPY_MAXDIMS];
        var dot; // PyArrray_DotFunc *dot
        var typec = null; // PyArray_Descr *typec = NULL;

        // make new pyarray object types from ops
        typenum = PyArray_ObjectType(op1, 0);
        typenum = PyArray_ObjectType(op2, typenum);

        // get type descriptor for the type, for us it is the javascript constructor
        // of the type, that we store as a dtype
        typec = PyArray_DescrFromType(typenum);

        // we currently do not support this specific check for common data type
        if (typec === null) {
            throw new Sk.builtin.ValueError('Cannot find a common data type.');
        }

        ap1 = PyArray_FromAny(op1, typec, 0, 0, 'NPY_ARRAY_ALINGED', null);
        ap2 = PyArray_FromAny(op2, typec, 0, 0, 'NPY_ARRAY_ALINGED', null);

        // check dimensions

        // handle 0 dim cases
        if (PyArray_NDIM(ap1) == 0 || PyArray_DIM(ap2) == 0) {
            ret = PyArray_NDIM(ap1) == 0 ? ap1 : ap2;
            ret = ret.nb$multiply(ap1, ap2);

            return ret;
        }

        l = PyArray_DIMS(ap1)[PyArray_NDIM(ap1) - 1];
        if (PyArray_NDIM(ap2) > 1) {
            matchDim = PyArray_NDIM(ap2) - 2;
        } else {
            matchDim = 0;
        }

        if (PyArray_DIMS(ap2)[matchDim] != l) {
            // dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, matchDim);
            throw new Sk.builtin.ValueError('shapes are not aligned');
        }

        nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
        if (nd > NPY_MAXDIMS) {
            throw new Sk.builtin.ValueError('dot: too many dimensions in result');
        }

        j = 0;
        for (i = 0; i < PyArray_NDIM(ap1) - 1; i++) {
            dimensions[j++] = PyArray_DIMS(ap1)[i];
        }
        for (i = 0; i < PyArray_NDIM(ap2) - 2; i++) {
            dimensions[j++] = PyArray_DIMS(ap2)[i];
        }
        if(PyArray_NDIM(ap2) > 1) {
            dimensions[j++] = PyArray_DIMS(ap2)[PyArray_NDIM(ap2)-1];
        }

        is1 = PyArray_STRIDES(ap1)[PyArray_NDIM(ap1)-1];
        is2 = PyArray_STRIDES(ap2)[matchDim];
        /* Choose which subtype to return */
        //ret = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum);

        // create new empty array for given dimensions
        ret = Sk.misceval.callsim(mod.zeros, dimensions, typec);

        // ToDo: check if both types allow dot multiplication
        //dot = PyArray_DESCR(ret)->f->dotfunc;
        //if (dot == NULL) {
        //    PyErr_SetString(PyExc_ValueError,
        //                    "dot not available for this type");
        //    goto fail;
        //}


        return null;
    }

  var np = new numpy();

  var mod = {};

  // doc string for numpy module
  mod.__doc__ = new Sk.builtin.str('\nNumPy\n=====\n\nProvides\n  1. An array object of arbitrary homogeneous items\n  2. Fast mathematical operations over arrays\n  3. Linear Algebra, Fourier Transforms, Random Number Generation\n\nHow to use the documentation\n----------------------------\nDocumentation is available in two forms: docstrings provided\nwith the code, and a loose standing reference guide, available from\n`the NumPy homepage <http://www.scipy.org>`_.\n\nWe recommend exploring the docstrings using\n`IPython <http://ipython.scipy.org>`_, an advanced Python shell with\nTAB-completion and introspection capabilities.  See below for further\ninstructions.\n\nThe docstring examples assume that `numpy` has been imported as `np`::\n\n  >>> import numpy as np\n\nCode snippets are indicated by three greater-than signs::\n\n  >>> x = 42\n  >>> x = x + 1\n\nUse the built-in ``help`` function to view a function\'s docstring::\n\n  >>> help(np.sort)\n  ... # doctest: +SKIP\n\nFor some objects, ``np.info(obj)`` may provide additional help.  This is\nparticularly true if you see the line "Help on ufunc object:" at the top\nof the help() page.  Ufuncs are implemented in C, not Python, for speed.\nThe native Python help() does not know how to view their help, but our\nnp.info() function does.\n\nTo search for documents containing a keyword, do::\n\n  >>> np.lookfor(\'keyword\')\n  ... # doctest: +SKIP\n\nGeneral-purpose documents like a glossary and help on the basic concepts\nof numpy are available under the ``doc`` sub-module::\n\n  >>> from numpy import doc\n  >>> help(doc)\n  ... # doctest: +SKIP\n\nAvailable subpackages\n---------------------\ndoc\n    Topical documentation on broadcasting, indexing, etc.\nlib\n    Basic functions used by several sub-packages.\nrandom\n    Core Random Tools\nlinalg\n    Core Linear Algebra Tools\nfft\n    Core FFT routines\npolynomial\n    Polynomial tools\ntesting\n    Numpy testing tools\nf2py\n    Fortran to Python Interface Generator.\ndistutils\n    Enhancements to distutils with support for\n    Fortran compilers support and more.\n\nUtilities\n---------\ntest\n    Run numpy unittests\nshow_config\n    Show numpy build configuration\ndual\n    Overwrite certain functions with high-performance Scipy tools\nmatlib\n    Make everything matrices.\n__version__\n    Numpy version string\n\nViewing documentation using IPython\n-----------------------------------\nStart IPython with the NumPy profile (``ipython -p numpy``), which will\nimport `numpy` under the alias `np`.  Then, use the ``cpaste`` command to\npaste examples into the shell.  To see which functions are available in\n`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use\n``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow\ndown the list.  To view the docstring for a function, use\n``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view\nthe source code).\n\nCopies vs. in-place operation\n-----------------------------\nMost of the functions in `numpy` return a copy of the array argument\n(e.g., `np.sort`).  In-place versions of these functions are often\navailable as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.\nExceptions to this rule are documented.\n\n');

  /**
        Class for numpy.ndarray
    **/
  var CLASS_NDARRAY = "numpy.ndarray";

  function remapToJs_shallow(obj, shallow) {
    var _shallow = shallow || true;
    if (obj instanceof Sk.builtin.list) {
      if (!_shallow) {
        var ret = [];
        for (var i = 0; i < obj.v.length; ++i) {
          ret.push(Sk.ffi.remapToJs(obj.v[i]));
        }
        return ret;
      } else {
        return obj.v;
      }
    } else if (obj instanceof Sk.builtin.float_) {
      return Sk.builtin.asnum$nofloat(obj);
    } else {
      return Sk.ffi.remapToJs(obj);
    }
  }

  /**
        Unpacks in any form fo nested Lists
    **/
  function unpack(py_obj, buffer, state) {
    if (py_obj instanceof Sk.builtin.list || py_obj instanceof Sk.builtin.tuple) {
      var py_items = remapToJs_shallow(py_obj);
      state.level += 1;

      if (state.level > state.shape.length) {
        state.shape.push(py_items.length);
      }
      var i;
      var len = py_items.length;
      for (i = 0; i < len; i++) {
        unpack(py_items[i], buffer, state);
      }
      state.level -= 1;
    } else {
      buffer.push(py_obj);
    }
  }

  /**
   Computes the strides for columns and rows
  **/
  function computeStrides(shape) {
    var strides = shape.slice(0);
    strides.reverse();
    var prod = 1;
    var temp;
    for (var i = 0, len = strides.length; i < len; i++) {
      temp = strides[i];
      strides[i] = prod;
      prod *= temp;
    }

    return strides.reverse();
  }

  /**
    Computes the offset for the ndarray for given index and strides
    [1, ..., n]
  **/
  function computeOffset(strides, index) {
    var offset = 0;
    for (var k = 0, len = strides.length; k < len; k++) {
      offset += strides[k] * index[k];
    }
    return offset;
  }

  /**
    Calculates the size of the ndarray, dummy
    **/
  function prod(numbers) {
    var size = 1;
    var i;
    for (i = 0; i < numbers.length; i++) {
      size *= numbers[i];
    }
    return size;
  }

  /**
        Creates a string representation for given buffer and shape
        buffer is an ndarray
    **/
  function stringify(buffer, shape, dtype) {
    var emits = shape.map(function (x) {
      return 0;
    });
    var uBound = shape.length - 1;
    var idxLevel = 0;
    var str = "[";
    var i = 0;
    while (idxLevel !== -1) {
      if (emits[idxLevel] < shape[idxLevel]) {
        if (emits[idxLevel] !== 0) {
          str += ", ";
        }

        if (idxLevel < uBound) {
          str += "[";
          idxLevel += 1;
        } else {
          if (dtype === Sk.builtin.float_)
            str += Sk.ffi.remapToJs(Sk.builtin.str(new Sk.builtin.float_(buffer[
              i++])));
          else
            str += Sk.ffi.remapToJs(Sk.builtin.str(buffer[i++]));
          emits[idxLevel] += 1;
        }
      } else {
        emits[idxLevel] = 0;
        str += "]";
        idxLevel -= 1;
        if (idxLevel >= 0) {
          emits[idxLevel] += 1;
        }
      }
    }
    return str;
  }

  /*
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html?highlight=tolist#numpy.ndarray.tolist
  */
  function tolistrecursive(buffer, shape, strides, startdim, dtype) {
    var i, n, stride;
    var arr, item;

    /* Base case */
    if (startdim >= shape.length) {
        return buffer[0];
    }

    n = shape[startdim];
    stride = strides[startdim];

    arr = [];

    for (i = 0; i < n; i++) {
      item = tolistrecursive(buffer, shape, strides, startdim + 1, dtype);
      arr.push(item);

      buffer = buffer.slice(stride);
    }

    return new Sk.builtin.list(arr);
  }

  /**
     internal tolist interface
    **/
  function tolist(buffer, shape, strides, dtype) {
    var buffer_copy = buffer.slice(0);
    return tolistrecursive(buffer_copy, shape, strides, 0, dtype);
  }

  /**
    An array object represents a multidimensional, homogeneous array of fixed-size items.
    An associated data-type object describes the format of each element in the array
    (its byte-order, how many bytes it occupies in memory, whether it is an integer, a
    floating point number, or something else, etc.)

    Arrays should be constructed using array, zeros or empty (refer to the See Also
    section below). The parameters given here refer to a low-level method (ndarray(...)) for
    instantiating an array.

    For more information, refer to the numpy module and examine the the methods and
    attributes of an array.
  **/
  var ndarray_f = function ($gbl, $loc) {
    $loc.__init__ = new Sk.builtin.func(function (self, shape, dtype, buffer,
      offset, strides, order) {
      var ndarrayJs = {}; // js object holding the actual array
      ndarrayJs.shape = Sk.ffi.remapToJs(shape);

      ndarrayJs.strides = computeStrides(ndarrayJs.shape);
      ndarrayJs.dtype = dtype || Sk.builtin.none.none$;

      // allow any nested data structure
      if (buffer && buffer instanceof Sk.builtin.list) {
        ndarrayJs.buffer = buffer.v; // ToDo: change this to any sequence and iterate over objects?
      }

      self.v = ndarrayJs; // value holding the actual js object and array
      self.tp$name = CLASS_NDARRAY; // set class name
    });

    $loc._internalGenericGetAttr = Sk.builtin.object.prototype.GenericSetAttr;

    $loc.__getattr__ = new Sk.builtin.func(function (self, name) {
        if (name != null && (Sk.builtin.checkString(name) || typeof name === "string")) {
            var _name = name;

            // get javascript string
            if (Sk.builtin.checkString(name)) {
                _name = Sk.ffi.remapToJs(name);
            }

            switch (_name) {
            case 'ndmin':
                return new Sk.builtin.int_(PyArray_NDIM(self));
            case 'dtype':
                return self.v.dtype;
            case 'shape':
                return new Sk.builtin.tuple(PyArray_DIMS(self).map(
                  function (x) {
                    return new Sk.builtin.int_(x);
                  }));
            case 'strides':
                return new Sk.builtin.tuple(PyArray_STRIDES(self).map(
                  function (x) {
                    return new Sk.builtin.int_(x);
                  }));
            case 'size':
                return new Sk.builtin.int_(PyArray_SIZE(self));
            case 'data':
                return new Sk.builtin.list(PyArray_DATA(self));
            case 'T':
                if (PyArray_NDIM(self) < 2) {
                    return self;
                } else {
                    return Sk.misceval.callsim(self.transpose, self);
                }
            }
        }

        // if we have not returned yet, try the genericgetattr
        return Sk.misceval.callsim(self.__getattribute__, self, name);
    });

    // ToDo: setAttribute should be implemented, change of shape causes resize
    // ndmin cannot be set, etc...
    $loc.__setattr__ = new Sk.builtin.func(function (self, name, value) {
        if (name != null && (Sk.builtin.checkString(name) || typeof name === "string")) {
            var _name = name;

            // get javascript string
            if (Sk.builtin.checkString(name)) {
                _name = Sk.ffi.remapToJs(name);
            }

            switch (_name) {
                case 'shape':
                    // trigger reshape;
                    if (value instanceof Sk.builtin.tuple || value instanceof Sk.builtin.list || Sk.builtin.checkInt(value)) {
                        var js_shape = PyArray_UNPACK_SHAPE(self, value);

                        // ToDo: shape unpacking: seq/int => tuple (js array)
                        self.v.shape = js_shape;
                    } else {
                        throw new TypeError('expected sequence object with len >= 0 or a single integer');
                    }
                    return;
            }
        }

        // builtin: --> all is readonly (I am not happy with this)
        throw new Sk.builtin.AttributeError("'ndarray' object attribute '" + name + "' is readonly");
    });

    /*
      Return the array as a (possibly nested) list.

      Return a copy of the array data as a (nested) Python list. Data items are
      converted to the nearest compatible Python type.
    */
    $loc.tolist = new Sk.builtin.func(function (self) {
      var ndarrayJs = Sk.ffi.remapToJs(self);
      var list = tolist(ndarrayJs.buffer, ndarrayJs.shape, ndarrayJs.strides,
        ndarrayJs.dtype);

      return list;
    });

    $loc.reshape = new Sk.builtin.func(function (self, shape, order) {
        Sk.builtin.pyCheckArgs("reshape", arguments, 2, 3);
        var ndarrayJs = Sk.ffi.remapToJs(self);
        // if the first dim is -1, then the size in infered from the size
        // and the other dimensions
        var js_shape = PyArray_UNPACK_SHAPE(self, shape);

        // create python object for shape
        var py_shape = Sk.ffi.remapToPy(js_shape);
        return Sk.misceval.callsim(mod[CLASS_NDARRAY], py_shape, ndarrayJs.dtype,
        new Sk.builtin.list(ndarrayJs.buffer));
    });

    $loc.copy = new Sk.builtin.func(function (self, order) {
      Sk.builtin.pyCheckArgs("copy", arguments, 1, 2);
      var ndarrayJs = Sk.ffi.remapToJs(self);
      var buffer = ndarrayJs.buffer.map(function (x) {
        return x;
      });
      var shape = new Sk.builtin.tuplePy(ndarrayJs.shape.map(function (x) {
        return new Sk.builtin.int_(x);
      }));
      return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, ndarrayJs.dtype,
        new Sk.builtin.list(buffer));
    });

    /**
      Fill the array with a scalar value.
      Parameters: value: scalar
                    All elements of a will be assigned this value
    **/
    $loc.fill = new Sk.builtin.func(function (self, value) {
      Sk.builtin.pyCheckArgs("fill", arguments, 2, 2);
      var ndarrayJs = Sk.ffi.remapToJs(self);
      var buffer = ndarrayJs.buffer.map(function (x) {
        return x;
      });
      var i;
      for (i = 0; i < ndarrayJs.buffer.length; i++) {
        if (ndarrayJs.dtype) {
          ndarrayJs.buffer[i] = Sk.misceval.callsim(ndarrayJs.dtype,
            value);
        }
      }
    });

    $loc.__getslice__ = new Sk.builtin.func( function (self, start, stop) {
      Sk.builtin.pyCheckArgs( "[]", arguments, 3, 2 );
      var ndarrayJs = Sk.ffi.remapToJs( self );
      var _index; // current index
      var _buffer; // buffer as python type
      var buffer_internal; // buffer als js array
      var _stride; // stride
      var _shape; // shape as js
      var i;
      var _start;
      var _stop;

      if ( !Sk.builtin.checkInt( start ) && !( Sk.builtin.checkInt( stop ) || Sk.builtin.checkNone( stop ) || stop === undefined) ) {
        // support for slices e.g. [1,4] or [0,]

        _start = Sk.ffi.remapToJs(start);
        if(stop === undefined || Sk.builtin.checkNone( stop )) {
          _stop = ndarrayJs.buffer.length;
        } else {
          _stop = Sk.ffi.remapToJs(start);
        }

        if(_start < 0 || _stop < 0) {
          throw new Sk.builtin.IndexError('Use of negative indices is not supported.');
        }

        buffer_internal = [];
        _index = 0;
        for ( i = _start; i < _stop; i += 1 ) {
          buffer_internal[ _index++ ] = ndarrayJs.buffer[ i ];
        }
        _buffer = new Sk.builtin.list( buffer_internal );
        _shape = new Sk.builtin.tuple( [ buffer_internal.length ].map(
          function (x) {
            return new Sk.builtin.int_( x );
          } ) );
        return Sk.misceval.callsim(mod[ CLASS_NDARRAY ], _shape, undefined,
          _buffer );
      } else {
        throw new Sk.builtin.ValueError( 'Index "' + index +
          '" must be int' );
      }
    } );

    $loc.__setslice__ = new Sk.builtin.func( function (self, start, stop, value) {
      Sk.builtin.pyCheckArgs( "[]", arguments, 3, 2 );
      var ndarrayJs = Sk.ffi.remapToJs( self );
      var _index; // current index
      var _buffer; // buffer as python type
      var buffer_internal; // buffer als js array
      var _stride; // stride
      var _shape; // shape as js
      var i;
      var _start;
      var _stop;

      if ( !Sk.builtin.checkInt( start ) && !( Sk.builtin.checkInt( stop ) || Sk.builtin.checkNone( stop ) || stop === undefined) ) {
        // support for slices e.g. [1,4] or [0,]

        _start = Sk.ffi.remapToJs(start);
        if(stop === undefined || Sk.builtin.checkNone( stop )) {
          _stop = ndarrayJs.buffer.length;
        } else {
          _stop = Sk.ffi.remapToJs(start);
        }

        if(_start < 0 || _stop < 0) {
          throw new Sk.builtin.IndexError('Use of negative indices is not supported.');
        }

        for ( i = _start; i < _stop; i += 1 ) {
          ndarrayJs.buffer[computeOffset(ndarrayJs.strides, i)] = value;
        }
      } else {
        throw new Sk.builtin.ValueError( 'Index "' + index +
          '" must be int' );
      }
    } );

    $loc.__getitem__ = new Sk.builtin.func(function (self, index) {
      Sk.builtin.pyCheckArgs("[]", arguments, 2, 2);
      var ndarrayJs = Sk.ffi.remapToJs(self);
      var _index; // current index
      var _buffer; // buffer as python type
      var buffer_internal; // buffer als js array
      var _stride; // stride
      var _shape; // shape as js
      var i;

      // single index e.g. [3]
      if (Sk.builtin.checkInt(index)) {
        var offset = Sk.ffi.remapToJs(index);

        if (ndarrayJs.shape.length > 1) {
          _stride = ndarrayJs.strides[0];
          buffer_internal = [];
          _index = 0;

          for (i = offset * _stride, ubound = (offset + 1) * _stride; i <
            ubound; i++) {
            buffer_internal[_index++] = ndarrayJs.buffer[i];
          }

          _buffer = new Sk.builtin.list(buffer_internal);
          _shape = new Sk.builtin.tuple(Array.prototype.slice.call(
              ndarrayJs.shape,
              1)
            .map(function (x) {
              return new Sk.builtin.int_(x);
            }));
          return Sk.misceval.callsim(mod[CLASS_NDARRAY], _shape,
            undefined,
            _buffer);
        } else {
          if (offset >= 0 && offset < ndarrayJs.buffer.length) {
            return ndarrayJs.buffer[offset];
          } else {
            throw new Sk.builtin.IndexError("array index out of range");
          }
        }
      } else if (index instanceof Sk.builtin.tuple) {
        // index like [1,3]
        var keyJs = Sk.ffi.remapToJs(index);
        return ndarrayJs.buffer[computeOffset(ndarrayJs.strides, keyJs)];
      } else if (index instanceof Sk.builtin.slice) {
        // support for slices e.g. [1:4]
        var indices = index.indices();
        var start = typeof indices[0] !== 'undefined' ? indices[0] : 0;
        var stop = typeof indices[1] !== 'undefined' ? indices[1] :
          ndarrayJs
          .buffer.length;
        stop = stop > ndarrayJs.buffer.length ? ndarrayJs.buffer.length :
          stop;
        var step = typeof indices[2] !== 'undefined' ? indices[2] : 1;
        buffer_internal = [];
        _index = 0;
        if (step > 0) {
          for (i = start; i < stop; i += step) {
            buffer_internal[_index++] = ndarrayJs.buffer[i];
          }
        }
        _buffer = new Sk.builtin.list(buffer_internal);
        _shape = new Sk.builtin.tuple([buffer_internal.length].map(
          function (
            x) {
            return new Sk.builtin.int_(x);
          }));
        return Sk.misceval.callsim(mod[CLASS_NDARRAY], _shape, undefined,
          _buffer);
      } else {
        throw new Sk.builtin.ValueError('Index "' + index +
          '" must be int, slice or tuple');
      }
    });

    $loc.__setitem__ = new Sk.builtin.func(function (self, index, value) {
      var ndarrayJs = Sk.ffi.remapToJs(self);
      Sk.builtin.pyCheckArgs("[]", arguments, 3, 3);
      if (Sk.builtin.checkInt(index)) {
        var _offset = Sk.ffi.remapToJs(index);
        if (ndarrayJs.shape.length > 1) {
          var _value = Sk.ffi.remapToJs(value);
          var _stride = ndarrayJs.strides[0];
          var _index = 0;

          var _ubound = (_offset + 1) * _stride;
          var i;
          for (i = _offset * _stride; i < _ubound; i++) {
            ndarrayJs.buffer[i] = _value.buffer[_index++];
          }
        } else {
          if (_offset >= 0 && _offset < ndarrayJs.buffer.length) {
            ndarrayJs.buffer[_offset] = value;
          } else {
            throw new Sk.builtin.IndexError("array index out of range");
          }
        }
      } else if (index instanceof Sk.builtin.tuple) {
        _key = Sk.ffi.remapToJs(index);
        ndarrayJs.buffer[computeOffset(ndarrayJs.strides, _key)] = value;
      } else {
        throw new Sk.builtin.TypeError(
          'argument "index" must be int or tuple');
      }
    });

    $loc.__len__ = new Sk.builtin.func(function (self) {
      var ndarrayJs = Sk.ffi.remapToJs(self);
      return new Sk.builtin.int_(ndarrayJs.shape[0]);
    });

    $loc.__iter__ = new Sk.builtin.func(function (self) {
      var ndarrayJs = Sk.ffi.remapToJs(self);
      var ret = {
        tp$iter: function () {
          return ret;
        },
        $obj: ndarrayJs,
        $index: 0,
        tp$iternext: function () {
          if (ret.$index >= ret.$obj.buffer.length) return undefined;
          return ret.$obj.buffer[ret.$index++];
        }
      };
      return ret;
    });

    $loc.__str__ = new Sk.builtin.func(function (self) {
      var ndarrayJs = remapToJs_shallow(self, false);
      return new Sk.builtin.str(stringify(ndarrayJs.buffer,
        ndarrayJs.shape, ndarrayJs.dtype));
    });

    $loc.__repr__ = new Sk.builtin.func(function (self) {
      var ndarrayJs = Sk.ffi.remapToJs(self);
      return new Sk.builtin.str("array(" + stringify(ndarrayJs.buffer,
        ndarrayJs.shape, ndarrayJs.dtype) + ")");
    });

    /**
      Creates left hand side operations for given binary operator
    **/
    function makeNumericBinaryOpLhs(operation) {
      return function (self, other) {
        var lhs;
        var rhs;
        var buffer; // external
        var _buffer; // internal use
        var shape; // new shape of returned ndarray
        var i;


        var ndarrayJs = Sk.ffi.remapToJs(self);

        if (Sk.abstr.typeName(other) === CLASS_NDARRAY) {
          lhs = ndarrayJs.buffer;
          rhs = Sk.ffi.remapToJs(other)
            .buffer;
          _buffer = [];
          for (i = 0, len = lhs.length; i < len; i++) {
            //_buffer[i] = operation(lhs[i], rhs[i]);
            _buffer[i] = Sk.abstr.binary_op_(lhs[i], rhs[i], operation);
          }
        } else {
          lhs = ndarrayJs.buffer;
          _buffer = [];
          for (i = 0, len = lhs.length; i < len; i++) {
            _buffer[i] = Sk.abstr.numberBinOp(lhs[i], other, operation);
          }
        }

        // create return ndarray
        shape = new Sk.builtin.tuple(ndarrayJs.shape.map(function (x) {
          return new Sk.builtin.int_(x);
        }));
        buffer = new Sk.builtin.list(_buffer);
        return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, undefined,
          buffer);
      };
    }

    function makeNumericBinaryOpRhs(operation) {
      return function (self, other) {
        var ndarrayJs = Sk.ffi.remapToJs(self);
        var rhsBuffer = ndarrayJs.buffer;
        var _buffer = [];
        for (var i = 0, len = rhsBuffer.length; i < len; i++) {
          _buffer[i] = Sk.abstr.numberBinOp(other, rhsBuffer[i], operation);
        }
        var shape = new Sk.builtin.tuple(ndarrayJs.shape.map(function (x) {
          return new Sk.builtin.int_(x);
        }));
        buffer = new Sk.builtin.list(_buffer);
        return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, undefined, buffer);
      };
    }

    /*
      Applies given operation on each element of the ndarray.
    */
    function makeUnaryOp(operation) {
      return function (self) {
        var ndarrayJs = Sk.ffi.remapToJs(self);
        var _buffer = ndarrayJs.buffer.map(function (value) {
          return Sk.abstr.numberUnaryOp(value, operation);
        });
        var shape = new Sk.builtin.tuple(ndarrayJs.shape.map(function (x) {
          return new Sk.builtin.int_(x);
        }));
        buffer = new Sk.builtin.list(_buffer);
        return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, undefined, buffer);
      };
    }

    $loc.__add__ = new Sk.builtin.func(makeNumericBinaryOpLhs("Add"));
    $loc.__radd__ = new Sk.builtin.func(makeNumericBinaryOpRhs("Add"));

    $loc.__sub__ = new Sk.builtin.func(makeNumericBinaryOpLhs("Sub"));
    $loc.__rsub__ = new Sk.builtin.func(makeNumericBinaryOpRhs("Sub"));

    $loc.__mul__ = new Sk.builtin.func(makeNumericBinaryOpLhs("Mult"));
    $loc.__rmul__ = new Sk.builtin.func(makeNumericBinaryOpRhs("Mult"));

    $loc.__div__ = new Sk.builtin.func(makeNumericBinaryOpLhs("Div"));
    $loc.__rdiv__ = new Sk.builtin.func(makeNumericBinaryOpRhs("Div"));

    $loc.__mod__ = new Sk.builtin.func(makeNumericBinaryOpLhs("Mod"));
    $loc.__rmod__ = new Sk.builtin.func(makeNumericBinaryOpRhs("Mod"));

    $loc.__xor__ = new Sk.builtin.func(makeNumericBinaryOpLhs("BitXor"));
    $loc.__rxor__ = new Sk.builtin.func(makeNumericBinaryOpRhs("BitXor"));

    $loc.__lshift__ = new Sk.builtin.func(makeNumericBinaryOpLhs("LShift"));
    $loc.__rlshift__ = new Sk.builtin.func(makeNumericBinaryOpRhs("LShift"));

    $loc.__rshift__ = new Sk.builtin.func(makeNumericBinaryOpLhs("RShift"));
    $loc.__rrshift__ = new Sk.builtin.func(makeNumericBinaryOpRhs("RShift"));

    $loc.__pos__ = new Sk.builtin.func(makeUnaryOp("UAdd"));
    $loc.__neg__ = new Sk.builtin.func(makeUnaryOp("USub"));

    /**
     Simple pow implementation that faciliates the pow builtin
    **/
    $loc.__pow__ = new Sk.builtin.func(function (self, other) {
      Sk.builtin.pyCheckArgs("__pow__", arguments, 2, 2);
      var ndarrayJs = Sk.ffi.remapToJs(self);
      var _buffer = ndarrayJs.buffer.map(function (value) {
        return Sk.builtin.pow(value, other);
      });
      var shape = new Sk.builtin.tuple(ndarrayJs.shape.map(function (x) {
        return new Sk.builtin.int_(x);
      }));
      buffer = new Sk.builtin.list(_buffer);
      return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, undefined, buffer);
    });

    $loc.transpose = new Sk.builtin.func(function (self, axes) {
        // http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.transpose.html
        if (axes == null || Sk.builtin.checkNone(axes)) {
            // reverse order of the axes and return
            var py_shape = new Sk.builtin.tuple(self.v.shape.map(function (x) {
              return new Sk.builtin.int_(x);
            }));
            var py_buffer = new Sk.builtin.list(self.v.buffer);
            var ndarray_copy = Sk.misceval.callsim(mod[ CLASS_NDARRAY ], py_shape, undefined, py_buffer);
            var reversed_shape = ndarray_copy.v.shape.reverse(); // reverse the order
            return ndarray_copy;
        } else {
            throw new Sk.builtin.NotImplementedError('ndarray.tranpose does not support the axes parameter');
        }
    });

    // end of ndarray_f
  };

  mod[CLASS_NDARRAY] = Sk.misceval.buildClass(mod, ndarray_f,
    CLASS_NDARRAY, []);

  /**
   Trigonometric functions, all element wise
  **/
  mod.pi = Sk.builtin.float_(np.math ? np.math.PI : Math.PI);
  mod.e = Sk.builtin.float_(np.math ? np.math.E : Math.E);
  /**
  Trigonometric sine, element-wise.
  **/

  function callTrigonometricFunc(x, op) {
    var res;
    var num;
    if (x instanceof Sk.builtin.list || x instanceof Sk.builtin.tuple) {
      x = Sk.misceval.callsim(mod.array, x);
    }

    if (Sk.abstr.typeName(x) === CLASS_NDARRAY) {
      var ndarrayJs = Sk.ffi.remapToJs(x);

      var _buffer = ndarrayJs.buffer.map(function (value) {
        num = Sk.builtin.asnum$(value);
        res = op.call(null, num);
        return new Sk.builtin.float_(res);
      });

      var shape = new Sk.builtin.tuple(ndarrayJs.shape.map(function (x) {
        return new Sk.builtin.int_(x);
      }));

      buffer = new Sk.builtin.list(_buffer);
      return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, undefined, buffer);
    } else if (Sk.builtin.checkNumber(x)) {
      num = Sk.builtin.asnum$(x);
      res = op.call(null, num);
      return new Sk.builtin.float_(res);
    }

    throw new Sk.builtin.TypeError('Unsupported argument type for "x"');
  }

  // Sine, element-wise.
  var sin_f = function (x, out) {
    Sk.builtin.pyCheckArgs("sin", arguments, 1, 2);
    return callTrigonometricFunc(x, np.math ? np.math.sin : Math.sin);
  };
  sin_f.co_varnames = ['x', 'out'];
  sin_f.$defaults = [0, new Sk.builtin.list([])];
  mod.sin = new Sk.builtin.func(sin_f);

  // Hyperbolic sine, element-wise.
  var sinh_f = function (x, out) {
    Sk.builtin.pyCheckArgs("sinh", arguments, 1, 2);
    if (!np.math) throw new Sk.builtin.OperationError("sinh requires mathjs");
    return callTrigonometricFunc(x, np.math.sinh);
  };
  sinh_f.co_varnames = ['x', 'out'];
  sinh_f.$defaults = [0, new Sk.builtin.list([])];
  mod.sinh = new Sk.builtin.func(sinh_f);

  // Inverse sine, element-wise.
  var arcsin_f = function (x, out) {
    Sk.builtin.pyCheckArgs("arcsin", arguments, 1, 2);
    return callTrigonometricFunc(x, np.math ? np.math.asin : Math.asin);
  };
  arcsin_f.co_varnames = ['x', 'out'];
  arcsin_f.$defaults = [0, new Sk.builtin.list([])];
  mod.arcsin = new Sk.builtin.func(arcsin_f);

  // Cosine, element-wise.
  var cos_f = function (x, out) {
    Sk.builtin.pyCheckArgs("cos", arguments, 1, 2);
    return callTrigonometricFunc(x, np.math ? np.math.cos : Math.cos);
  };
  cos_f.co_varnames = ['x', 'out'];
  cos_f.$defaults = [0, new Sk.builtin.list([])];
  mod.cos = new Sk.builtin.func(cos_f);

  // Hyperbolic cosine, element-wise.
  var cosh_f = function (x, out) {
    Sk.builtin.pyCheckArgs("cosh", arguments, 1, 2);
    if (!np.math) throw new Sk.builtin.OperationError("cosh requires mathjs");
    return callTrigonometricFunc(x, np.math.cosh);
  };
  cosh_f.co_varnames = ['x', 'out'];
  cosh_f.$defaults = [0, new Sk.builtin.list([])];
  mod.cosh = new Sk.builtin.func(cosh_f);

  // Inverse cosine, element-wise.
  var arccos_f = function (x, out) {
    Sk.builtin.pyCheckArgs("arccos", arguments, 1, 2);
    return callTrigonometricFunc(x, np.math ? np.math.acos : Math.acos);
  };
  arccos_f.co_varnames = ['x', 'out'];
  arccos_f.$defaults = [0, new Sk.builtin.list([])];
  mod.arccos = new Sk.builtin.func(arccos_f);

  // Inverse tangens, element-wise.
  var arctan_f = function (x, out) {
    Sk.builtin.pyCheckArgs("arctan", arguments, 1, 2);
    return callTrigonometricFunc(x, np.math ? np.math.atan : Math.atan);
  };
  arctan_f.co_varnames = ['x', 'out'];
  arctan_f.$defaults = [0, new Sk.builtin.list([])];
  mod.arctan = new Sk.builtin.func(arctan_f);

  // Tangens, element-wise.
  var tan_f = function (x, out) {
    Sk.builtin.pyCheckArgs("tan", arguments, 1, 2);
    return callTrigonometricFunc(x, np.math ? np.math.tan : Math.tan);
  };
  tan_f.co_varnames = ['x', 'out'];
  tan_f.$defaults = [0, new Sk.builtin.list([])];
  mod.tan = new Sk.builtin.func(tan_f);

  // Hyperbolic cosine, element-wise.
  var tanh_f = function (x, out) {
    Sk.builtin.pyCheckArgs("tanh", arguments, 1, 2);
    if (!np.math) throw new Sk.builtin.OperationError("tanh requires mathjs");
    return callTrigonometricFunc(x, np.math.tanh);
  };
  tanh_f.co_varnames = ['x', 'out'];
  tanh_f.$defaults = [0, new Sk.builtin.list([])];
  mod.tanh = new Sk.builtin.func(tanh_f);


  // Exponential
  var exp_f = function (x, out) {
    Sk.builtin.pyCheckArgs("exp", arguments, 1, 2);

    /* for complex type support we should use here a different approach*/
    //Sk.builtin.assk$(Math.E, Sk.builtin.nmber.float$);
    return callTrigonometricFunc(x, np.math ? np.math.exp : Math.exp);
  };
  exp_f.co_varnames = ['x', 'out'];
  exp_f.$defaults = [0, new Sk.builtin.list([])];
  mod.exp = new Sk.builtin.func(exp_f);

  // Square Root
  var sqrt_f = function (x, out) {
    Sk.builtin.pyCheckArgs("sqrt", arguments, 1, 2);
   return callTrigonometricFunc(x, np.math ? np.math.sqrt : Math.sqrt);
  };
  sqrt_f.co_varnames = ['x', 'out'];
  sqrt_f.$defaults = [0, new Sk.builtin.list([])];
  mod.sqrt = new Sk.builtin.func(sqrt_f);


  /* Simple reimplementation of the linspace function
   * http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
   */
  var linspace_f = function (start, stop, num, endpoint, retstep) {
    Sk.builtin.pyCheckArgs("linspace", arguments, 3, 5);
    Sk.builtin.pyCheckType("start", "number", Sk.builtin.checkNumber(
      start));
    Sk.builtin.pyCheckType("stop", "number", Sk.builtin.checkNumber(
      stop));
    if (num === undefined) {
      num = 50;
    }
    var num_num = Sk.builtin.asnum$(num);
    var endpoint_bool;

    if (endpoint === undefined) {
      endpoint_bool = true;
    } else if (endpoint.constructor === Sk.builtin.bool) {
      endpoint_bool = endpoint.v;
    }

    var retstep_bool;
    if (retstep === undefined) {
      retstep_bool = false;
    } else if (retstep.constructor === Sk.builtin.bool) {
      retstep_bool = retstep.v;
    }

    var samples;
    var step;

    start_num = Sk.builtin.asnum$(start) * 1.0;
    stop_num = Sk.builtin.asnum$(stop) * 1.0;

    if (num_num <= 0) {
      samples = [];
    } else {

      var samples_array;
      if (endpoint_bool) {
        if (num_num == 1) {
          samples = [start_num];
        } else {
          step = (stop_num - start_num) / (num_num - 1);
          samples_array = np.arange(0, num_num);
          samples = samples_array.map(function (v) {
            return v * step + start_num;
          });
          samples[samples.length - 1] = stop_num;
        }
      } else {
        step = (stop_num - start_num) / num_num;
        samples_array = np.arange(0, num_num);
        samples = samples_array.map(function (v) {
          return v * step + start_num;
        });
      }
    }

    //return as ndarray! dtype:float
    var dtype = Sk.builtin.float_;
    for (i = 0; i < samples.length; i++) {
      samples[i] = Sk.misceval.callsim(dtype, samples[i]);
    }

    var buffer = Sk.builtin.list(samples);
    var shape = new Sk.builtin.tuple([samples.length]);
    var ndarray = Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, dtype,
      buffer);

    if (retstep_bool === true)
      return new Sk.builtin.tuple([ndarray, step]);
    else
      return ndarray;
  };

  // this should allow for named parameters
  linspace_f.co_varnames = ['start', 'stop', 'num', 'endpoint',
    'retstep'
  ];
  linspace_f.$defaults = [0, 0, 50, true, false];
  mod.linspace =
    new Sk.builtin.func(linspace_f);

  /* Simple reimplementation of the arange function
   * http://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html#numpy.arange
   */
  var arange_f = function (start, stop, step, dtype) {
    Sk.builtin.pyCheckArgs("arange", arguments, 1, 4);
    Sk.builtin.pyCheckType("start", "number", Sk.builtin.checkNumber(
      start));
    var start_num;
    var stop_num;
    var step_num;

    if (stop === undefined && step === undefined) {
      start_num = Sk.builtin.asnum$(0);
      stop_num = Sk.builtin.asnum$(start);
      step_num = Sk.builtin.asnum$(1);
    } else if (step === undefined) {
      start_num = Sk.builtin.asnum$(start);
      stop_num = Sk.builtin.asnum$(stop);
      step_num = Sk.builtin.asnum$(1);
    } else {
      start_num = Sk.builtin.asnum$(start);
      stop_num = Sk.builtin.asnum$(stop);
      step_num = Sk.builtin.asnum$(step);
    }

    // set to float
    if (!dtype || dtype == Sk.builtin.none.none$) {
      if (Sk.builtin.checkInt(start))
        dtype = Sk.builtin.int_;
      else
        dtype = Sk.builtin.float_;
    }

    // return ndarray
    var arange_buffer = np.arange(start_num, stop_num, step_num);
    // apply dtype casting function, if it has been provided
    if (dtype && Sk.builtin.checkClass(dtype)) {
      for (i = 0; i < arange_buffer.length; i++) {
        arange_buffer[i] = Sk.misceval.callsim(dtype, arange_buffer[i]);
      }
    }

    buffer = Sk.builtin.list(arange_buffer);
    var shape = new Sk.builtin.tuple([arange_buffer.length]);
    return Sk.misceval.callsim(mod[CLASS_NDARRAY], shape, dtype,
      buffer);
  };

  arange_f.co_varnames = ['start', 'stop', 'step', 'dtype'];
  arange_f
    .$defaults = [0, 1, 1, Sk.builtin.none.none$];
  mod.arange = new Sk.builtin
    .func(arange_f);

  /* implementation for numpy.array
    ------------------------------------------------------------------------------------------------
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html#numpy.array

        object : array_like
        An array, any object exposing the array interface, an object whose __array__ method returns an array, or any (nested) sequence.

        dtype : data-type, optional
        The desired data-type for the array. If not given, then the type will be determined as the minimum type required to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use the .astype(t) method.

        copy : bool, optional
        If true (default), then the object is copied. Otherwise, a copy will only be made if __array__ returns a copy, if obj is a nested sequence, or if a copy is needed to satisfy any of the other requirements (dtype, order, etc.).

        order : {‘C’, ‘F’, ‘A’}, optional
        Specify the order of the array. If order is ‘C’ (default), then the array will be in C-contiguous order (last-index varies the fastest). If order is ‘F’, then the returned array will be in Fortran-contiguous order (first-index varies the fastest). If order is ‘A’, then the returned array may be in any order (either C-, Fortran-contiguous, or even discontiguous).

        subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned array will be forced to be a base-class array (default).

        ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have. Ones will be pre-pended to the shape as needed to meet this requirement.

        Returns :
        out : ndarray
        An array object satisfying the specified requirements
    */
  // https://github.com/geometryzen/davinci-dev/blob/master/src/stdlib/numpy.js
  // https://github.com/geometryzen/davinci-dev/blob/master/src/ffh.js
  // http://docs.scipy.org/doc/numpy/reference/arrays.html
  var array_f = function (object, dtype, copy, order, subok, ndmin) {
    Sk.builtin.pyCheckArgs("array", arguments, 1, 6);

    // ToDo: use PyArray_FromAny here and then do some checkings for the type
    // and maybe casting and support ndmin param
    // see http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html#numpy.array

    if (object === undefined)
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(object) +
        "' object is undefined");

    // check for ndmin param
    if (ndmin != null && Sk.builtin.checkInt(ndmin) === false) {
      throw new Sk.builtin.TypeError('Parameter "ndmin" must be of type "int"');
    }

    var py_ndarray = PyArray_FromAny(object, dtype, ndmin);

    return py_ndarray;
  };

  array_f.co_varnames = ['object', 'dtype', 'copy', 'order',
    'subok', 'ndmin'
  ];
  array_f.$defaults = [null, Sk.builtin.none.none$, true, new Sk.builtin.str(
    'C'), false, new Sk.builtin.int_(0)];
  mod.array = new Sk.builtin.func(array_f);

  /**
    Return a new array of given shape and type, filled with zeros.
  **/
  var zeros_f = function (shape, dtype, order) {
    Sk.builtin.pyCheckArgs("zeros", arguments, 1, 3);
    if(!Sk.builtin.checkSequence(shape) && !Sk.builtin.checkInt(shape)) {
      throw new Sk.builtin.TypeError('argument "shape" must int or sequence of ints');
    }

    if (dtype instanceof Sk.builtin.list) {
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(dtype) +
        "' is not supported for dtype.");
    }

    var _zero = new Sk.builtin.float_(0.0);

    return Sk.misceval.callsim(mod.full, shape, _zero, dtype, order);
  };
  zeros_f.co_varnames = ['shape', 'dtype', 'order'];
  zeros_f.$defaults = [
    new Sk.builtin.tuple([]), Sk.builtin.none.none$, new Sk.builtin.str('C')
  ];
  mod.zeros = new Sk.builtin.func(zeros_f);

  /**
    Return a new array of given shape and type, filled with `fill_value`.
  **/
  var full_f = function (shape, fill_value, dtype, order) {
    Sk.builtin.pyCheckArgs("full", arguments, 2, 4);

    if(!Sk.builtin.checkSequence(shape) && !Sk.builtin.checkInt(shape)) {
      throw new Sk.builtin.TypeError('argument "shape" must int or sequence of ints');
    }

    if (dtype instanceof Sk.builtin.list) {
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(dtype) +
        "' is currently not supported for dtype.");
    }

    var _shape = Sk.ffi.remapToJs(shape);
    var _tup;
    if(Sk.builtin.checkInt(shape)) {
      _tup = [];
      _tup.push(_shape);
      _shape = _tup;
    }
      // generate an array of the dimensions for the generic array method

    var _size = prod(_shape);
    var _buffer = [];
    var _fill_value = fill_value;
    var i;

    for (i = 0; i < _size; i++) {
      _buffer[i] = _fill_value;
    }

    // if no dtype given and type of fill_value is numeric, assume float
    if (!dtype && Sk.builtin.checkNumber(fill_value)) {
      dtype = Sk.builtin.float_;
    }

    // apply dtype casting function, if it has been provided
    if (Sk.builtin.checkClass(dtype)) {
      for (i = 0; i < _buffer.length; i++) {
        if (_buffer[i] !== Sk.builtin.none.none$) {
          _buffer[i] = Sk.misceval.callsim(dtype, _buffer[i]);
        }
      }
    }
    return Sk.misceval.callsim(mod[CLASS_NDARRAY], new Sk.builtin.list(_shape), dtype, new Sk.builtin
      .list(
        _buffer));
  };
  full_f.co_varnames = ['shape', 'fill_value', 'dtype', 'order'];
  full_f.$defaults = [
    new Sk.builtin.tuple([]), Sk.builtin.none.none$, Sk.builtin.none.none$, new Sk
    .builtin
    .str('C')
  ];
  mod.full = new Sk.builtin.func(full_f);


  /**
    Return a new array of given shape and type, filled with ones.
  **/
  var ones_f = function (shape, dtype, order) {
    Sk.builtin.pyCheckArgs("ones", arguments, 1, 3);

    if(!Sk.builtin.checkSequence(shape) && !Sk.builtin.checkInt(shape)) {
      throw new Sk.builtin.TypeError('argument "shape" must int or sequence of ints');
    }

    if (dtype instanceof Sk.builtin.list) {
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(dtype) +
        "' is not supported for dtype.");
    }

    var _one = new Sk.builtin.float_(1.0);
    return Sk.misceval.callsim(mod.full, shape, _one, dtype, order);
  };
  ones_f.co_varnames = ['shape', 'dtype', 'order'];
  ones_f.$defaults = [
    new Sk.builtin.tuple([]), Sk.builtin.none.none$, new Sk.builtin.str('C')
  ];
  mod.ones = new Sk.builtin.func(ones_f);

  /**
    Return a new array of given shape and type, filled with None.
  **/
  var empty_f = function (shape, dtype, order) {
    Sk.builtin.pyCheckArgs("empty", arguments, 1, 3);

    if(!Sk.builtin.checkSequence(shape) && !Sk.builtin.checkInt(shape)) {
      throw new Sk.builtin.TypeError('argument "shape" must int or sequence of ints');
    }

    if (dtype instanceof Sk.builtin.list) {
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(dtype) +
        "' is not supported for dtype.");
    }

    var _empty = Sk.builtin.none.none$;
    return Sk.misceval.callsim(mod.full, shape, _empty, dtype, order);
  };
  empty_f.co_varnames = ['shape', 'dtype', 'order'];
  empty_f.$defaults = [
    new Sk.builtin.tuple([]), Sk.builtin.none.none$, new Sk.builtin.str('C')
  ];
  mod.empty = new Sk.builtin.func(empty_f);

  /**
    Dot product
  **/
  var dot_f = function (a, b) {
    Sk.builtin.pyCheckArgs("dot", arguments, 2, 2);

    // ToDo: add support for ndarray args

    if (!(a instanceof Sk.builtin.list) && !Sk.builtin.checkNumber(
      a) && (Sk.abstr.typeName(a) !== CLASS_NDARRAY)) {
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(a) +
        "' is not supported for a.");
    }

    if (!(b instanceof Sk.builtin.list) && !Sk.builtin.checkNumber(
      b) && (Sk.abstr.typeName(b) !== CLASS_NDARRAY)) {
      throw new Sk.builtin.TypeError("'" + Sk.abstr.typeName(b) +
        "' is not supported for b.");
    }

    var res;

    var b_matrix;
    var a_matrix;

    if (Sk.abstr.typeName(a) === CLASS_NDARRAY) {
      a_matrix = a.v.buffer;
            a_matrix = a_matrix.map(function (x) {
                return Sk.ffi.remapToJs(x);
            });
    } else {
      a_matrix = Sk.ffi.remapToJs(a);
    }

    if (Sk.abstr.typeName(b) === CLASS_NDARRAY) {
      b_matrix = b.v.buffer;
            b_matrix = b_matrix.map(function (x) {
                return Sk.ffi.remapToJs(x);
            });
    } else {
      b_matrix = Sk.ffi.remapToJs(b);
    }

    var a_size = np.math.size(a_matrix);
    var b_size = np.math.size(b_matrix);

    // check for correct dimensions

    if (a_size.length >= 1 && b_size.length > 1) {
      if (a_size[a_size.length - 1] != b_size[b_size - 2]) {
        throw new Sk.builtin.ValueError(
          "The last dimension of a is not the same size as the second-to-last dimension of b."
        );
      }
    }

    res = np.math.multiply(a_matrix, b_matrix);

        if (!Array.isArray(res)) { // if result
            return Sk.ffi.remapToPy(res);
        }

    // return ndarray
    buffer = new Sk.builtin.list(res);
    return Sk.misceval.callsim(mod.array, buffer, Sk.builtin.float_);
  };
  dot_f.co_varnames = ['a', 'b'];
  dot_f.$defaults = [Sk.builtin.none.none$,
    Sk.builtin.none.none$
  ];
  mod.dot = new Sk.builtin.func(dot_f);

  // https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/multiarraymodule.c#L2252
  var vdot_f = function(a, b) {
        // a and b must be array like
        // if a or b have more than 1 dim => flatten them
        var typenum; // int
        var ip1;
        var ip2;
        var op;
        var op1 = a;
        var op2 = b;
        var newdimptr = -1;
        var newdims = [-1, 1];
        var ap1 = null;
        var ap2 = null;
        var ret = null;
        var type;
        var vdot;

        typenum = PyArray_ObjectType(op1, 0);
        typenum = PyArray_ObjectType(op2, typenum);

        type = PyArray_DescrFromType(typenum);

        ap1 = PyArray_FromAny(op1, type, 0, 0, 0, null);
        if (ap1 == null) {
            return null;
        }

        // flatten the array
        op1 = PyArray_NewShape(ap1, newdims, 'NPY_CORDER');
        if (op1 == null) {
            return;
        }
        ap1 = op1;

        ap2 = PyArray_FromAny(op2, type, 0, 0, 0, null);
        if (ap2 == null) {
            return null;
        }

        // flatten the array
        op2 = PyArray_NewShape(ap2, newdims, 'NPY_CORDER');
        if (op2 == null) {
            return;
        }
        ap2 = op2;

        if (PyArray_DIM(ap2, 0) != PyArray_DIM(ap1, 0)) {
            throw new Sk.builtin.ValueError('vectors have different lengths');
        }

        var shape = new Sk.builtin.tuple([0].map(
          function (x) {
            return new Sk.builtin.int_(x);
        }));
        // create new empty array for given dimensions
        ret = Sk.misceval.callsim(mod.zeros, shape, type);

        n = PyArray_DIM(ap1, 0);
        stride1 = PyArray_STRIDE(ap1, 0);
        stride2 = PyArray_STRIDE(ap2, 0);
        ip1 = PyArray_DATA(ap1);
        ip2 = PyArray_DATA(ap2);
        op = PyArray_DATA(ret);

        switch(typenum) {
        case 0:
        case 1:
        case 2:
        case 3:
            vdot = OBJECT_vdot;
            break;
        default:
            throw new Sk.builtin.ValueError('function not available for this data type');
        }

        // call vdot function with vectors
        vdot.call(null, ip1, stride1, ip2, stride2, op, n, null);

        // return resulting ndarray
        return PyArray_Return(ret);
  }
  mod.vdot = new Sk.builtin.func(vdot_f);

  /* not implemented methods */
  mod.ones_like = new Sk.builtin.func(function () {
    throw new Sk.builtin.NotImplementedError(
      "ones_like is not yet implemented");
  });
  mod.empty_like = new Sk.builtin.func(function () {
    throw new Sk.builtin.NotImplementedError(
      "empty_like is not yet implemented");
  });
  mod.ones_like = new Sk.builtin.func(function () {
    throw new Sk.builtin.NotImplementedError(
      "ones_like is not yet implemented");
  });
  mod.arctan2 = new Sk.builtin.func(function () {
    throw new Sk.builtin.NotImplementedError(
      "arctan2 is not yet implemented");
  });
  mod.asarray = new Sk.builtin.func(array_f);
  return mod;
};
