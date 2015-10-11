import unittest
import numpy as np
import numpy.random as random

class TestDot(unittest.TestCase):
    def setUp(self):
        np.random.seed(128)
        self.A = np.random.rand(4, 2)
        self.b1 = np.random.rand(2, 1)
        self.b2 = np.random.rand(2)
        self.b3 = np.random.rand(1, 2)
        self.b4 = np.random.rand(4)
        self.N = 7

    def test_dotmatmat(self):
        A = self.A
        res = np.dot(A.transpose(), A)
        tgt = np.array([[1.45046013, 0.86323640],
                        [0.86323640, 0.84934569]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotmatvec(self):
        A, b1 = self.A, self.b1
        res = np.dot(A, b1)
        tgt = np.array([[0.32114320], [0.04889721],
                        [0.15696029], [0.33612621]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotmatvec2(self):
        A, b2 = self.A, self.b2
        res = np.dot(A, b2)
        tgt = np.array([0.29677940, 0.04518649, 0.14468333, 0.31039293])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat(self):
        A, b4 = self.A, self.b4
        res = np.dot(b4, A)
        tgt = np.array([1.23495091, 1.12222648])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat2(self):
        b3, A = self.b3, self.A
        res = np.dot(b3, A.transpose())
        tgt = np.array([[0.58793804, 0.08957460, 0.30605758, 0.62716383]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat3(self):
        A, b4 = self.A, self.b4
        res = np.dot(A.transpose(), b4)
        tgt = np.array([1.23495091, 1.12222648])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecvecouter(self):
        b1, b3 = self.b1, self.b3
        res = np.dot(b1, b3)
        tgt = np.array([[0.20128610, 0.08400440], [0.07190947, 0.03001058]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecvecinner(self):
        b1, b3 = self.b1, self.b3
        res = np.dot(b3, b1)
        tgt = np.array([[ 0.23129668]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotcolumnvect1(self):
        b1 = np.ones((3, 1))
        b2 = [5.3]
        res = np.dot(b1, b2)
        tgt = np.array([5.3, 5.3, 5.3])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotcolumnvect2(self):
        b1 = np.ones((3, 1)).transpose()
        b2 = [6.2]
        res = np.dot(b2, b1)
        tgt = np.array([6.2, 6.2, 6.2])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecscalar(self):
        np.random.seed(100)
        b1 = np.random.rand(1, 1)
        b2 = np.random.rand(1, 4)
        res = np.dot(b1, b2)
        tgt = np.array([[0.15126730, 0.23068496, 0.45905553, 0.00256425]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecscalar2(self):
        np.random.seed(100)
        b1 = np.random.rand(4, 1)
        b2 = np.random.rand(1, 1)
        res = np.dot(b1, b2)
        tgt = np.array([[0.00256425],[0.00131359],[0.00200324],[ 0.00398638]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_all(self):
        dims = [(), (1,), (1, 1)]
        dout = [(), (1,), (1, 1), (1,), (), (1,), (1, 1), (1,), (1, 1)]
        for dim, (dim1, dim2) in zip(dout, itertools.product(dims, dims)):
            b1 = np.zeros(dim1)
            b2 = np.zeros(dim2)
            res = np.dot(b1, b2)
            tgt = np.zeros(dim)
            assert_(res.shape == tgt.shape)
            assert_almost_equal(res, tgt, decimal=self.N)

    def test_vecobject(self):
        class Vec(object):
            def __init__(self, sequence=None):
                if sequence is None:
                    sequence = []
                self.array = np.array(sequence)

            def __add__(self, other):
                out = Vec()
                out.array = self.array + other.array
                return out

            def __sub__(self, other):
                out = Vec()
                out.array = self.array - other.array
                return out

            def __mul__(self, other):  # with scalar
                out = Vec(self.array.copy())
                out.array *= other
                return out

            def __rmul__(self, other):
                return self*other

        U_non_cont = np.transpose([[1., 1.], [1., 2.]])
        U_cont = np.ascontiguousarray(U_non_cont)
        x = np.array([Vec([1., 0.]), Vec([0., 1.])])
        zeros = np.array([Vec([0., 0.]), Vec([0., 0.])])
        zeros_test = np.dot(U_cont, x) - np.dot(U_non_cont, x)
        assert_equal(zeros[0].array, zeros_test[0].array)
        assert_equal(zeros[1].array, zeros_test[1].array)

    def test_dot_2args(self):
        from numpy.core.multiarray import dot

        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1, 0], [1, 1]], dtype=float)
        c = np.array([[3, 2], [7, 4]], dtype=float)

        d = dot(a, b)
        assert_allclose(c, d)

    def test_dot_3args(self):
        from numpy.core.multiarray import dot

        np.random.seed(22)
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))

        r = np.empty((1024, 32))
        for i in range(12):
            dot(f, v, r)
        assert_equal(sys.getrefcount(r), 2)
        r2 = dot(f, v, out=None)
        assert_array_equal(r2, r)
        assert_(r is dot(f, v, out=r))

        v = v[:, 0].copy()  # v.shape == (16,)
        r = r[:, 0].copy()  # r.shape == (1024,)
        r2 = dot(f, v)
        assert_(r is dot(f, v, r))
        assert_array_equal(r2, r)

    def test_dot_3args_errors(self):
        from numpy.core.multiarray import dot

        np.random.seed(22)
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))

        r = np.empty((1024, 31))
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((1024,))
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((32,))
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((32, 1024))
        assert_raises(ValueError, dot, f, v, r)
        assert_raises(ValueError, dot, f, v, r.T)

        r = np.empty((1024, 64))
        assert_raises(ValueError, dot, f, v, r[:, ::2])
        assert_raises(ValueError, dot, f, v, r[:, :32])

        r = np.empty((1024, 32), dtype=np.float32)
        assert_raises(ValueError, dot, f, v, r)

        r = np.empty((1024, 32), dtype=int)
        assert_raises(ValueError, dot, f, v, r)

    def test_dot_array_order(self):
        a = np.array([[1, 2], [3, 4]], order='C')
        b = np.array([[1, 2], [3, 4]], order='F')
        res = np.dot(a, a)

        # integer arrays are exact
        assert_equal(np.dot(a, b), res)
        assert_equal(np.dot(b, a), res)
        assert_equal(np.dot(b, b), res)

    def test_dot_scalar_and_matrix_of_objects(self):
        # Ticket #2469
        arr = np.matrix([1, 2], dtype=object)
        desired = np.matrix([[3, 6]], dtype=object)
        assert_equal(np.dot(arr, 3), desired)
        assert_equal(np.dot(3, arr), desired)

    def test_dot_override(self):
        class A(object):
            def __numpy_ufunc__(self, ufunc, method, pos, inputs, **kwargs):
                return "A"

        class B(object):
            def __numpy_ufunc__(self, ufunc, method, pos, inputs, **kwargs):
                return NotImplemented

        a = A()
        b = B()
        c = np.array([[1]])

        assert_equal(np.dot(a, b), "A")
        assert_equal(c.dot(a), "A")
        assert_raises(TypeError, np.dot, b, c)
        assert_raises(TypeError, c.dot, b)

    def test_accelerate_framework_sgemv_fix(self):

        def aligned_array(shape, align, dtype, order='C'):
            d = dtype(0)
            N = np.prod(shape)
            tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
            address = tmp.__array_interface__["data"][0]
            for offset in range(align):
                if (address + offset) % align == 0:
                    break
            tmp = tmp[offset:offset+N*d.nbytes].view(dtype=dtype)
            return tmp.reshape(shape, order=order)

        def as_aligned(arr, align, dtype, order='C'):
            aligned = aligned_array(arr.shape, align, dtype, order)
            aligned[:] = arr[:]
            return aligned

        def assert_dot_close(A, X, desired):
            assert_allclose(np.dot(A, X), desired, rtol=1e-5, atol=1e-7)

        m = aligned_array(100, 15, np.float32)
        s = aligned_array((100, 100), 15, np.float32)
        np.dot(s, m)  # this will always segfault if the bug is present

        testdata = itertools.product((15,32), (10000,), (200,89), ('C','F'))
        for align, m, n, a_order in testdata:
            # Calculation in double precision
            A_d = np.random.rand(m, n)
            X_d = np.random.rand(n)
            desired = np.dot(A_d, X_d)
            # Calculation with aligned single precision
            A_f = as_aligned(A_d, align, np.float32, order=a_order)
            X_f = as_aligned(X_d, align, np.float32)
            assert_dot_close(A_f, X_f, desired)
            # Strided A rows
            A_d_2 = A_d[::2]
            desired = np.dot(A_d_2, X_d)
            A_f_2 = A_f[::2]
            assert_dot_close(A_f_2, X_f, desired)
            # Strided A columns, strided X vector
            A_d_22 = A_d_2[:, ::2]
            X_d_2 = X_d[::2]
            desired = np.dot(A_d_22, X_d_2)
            A_f_22 = A_f_2[:, ::2]
            X_f_2 = X_f[::2]
            assert_dot_close(A_f_22, X_f_2, desired)
            # Check the strides are as expected
            if a_order == 'F':
                assert_equal(A_f_22.strides, (8, 8 * m))
            else:
                assert_equal(A_f_22.strides, (8 * n, 8))
            assert_equal(X_f_2.strides, (8,))
            # Strides in A rows + cols only
            X_f_2c = as_aligned(X_f_2, align, np.float32)
            assert_dot_close(A_f_22, X_f_2c, desired)
            # Strides just in A cols
            A_d_12 = A_d[:, ::2]
            desired = np.dot(A_d_12, X_d_2)
            A_f_12 = A_f[:, ::2]
            assert_dot_close(A_f_12, X_f_2c, desired)
            # Strides in A cols and X
            assert_dot_close(A_f_12, X_f_2, desired)

