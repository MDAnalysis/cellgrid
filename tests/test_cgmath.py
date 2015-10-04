import numpy as np
from numpy.testing import assert_array_almost_equal
import itertools

from cellgrid import cgmath


class TestInterDistance(object):
    def _manual(self, a, b):
        c = a - b[:,None]
        c = np.sqrt((c * c).sum(axis=-1))
        return np.ravel(c)

    def test_1(self):
        a = np.arange(30).reshape(10, 3).astype(np.float32)
        b = np.arange(30).reshape(10, 3).astype(np.float32)
        res = np.zeros(100).astype(np.float32)

        cgmath.inter_distance_array(a, b, res)

        ref = self._manual(a, b)

        assert_array_almost_equal(ref, res)

class TestIntraDistance(object):
    def _manual(self, a):
        def dist(x, y):
            v = x - y
            return np.sqrt(np.dot(v, v))

        n = a.shape[0]
        ref = np.zeros(n*(n-1)/2)

        pos = 0
        for i in range(n):
            for j in range(i+1, n):
                ref[pos] = dist(a[i], a[j])
                pos += 1

        return ref

    def test_1(self):
        a = np.arange(30).reshape(10, 3).astype(np.float32)
        n = 10
        res = np.zeros(n * (n-1) // 2, dtype=np.float32)

        cgmath.intra_distance_array(a, res)

        ref = self._manual(a)

        assert_array_almost_equal(ref, res)


class TestInterIndex(object):
    def _manual(self, a, b):
        ref = np.zeros((a.shape[0] * b.shape[0], 2), dtype=np.int)

        ref[:,0] = np.repeat(a, b.shape[0])
        ref[:,1] = list(b) * a.shape[0]

        return ref

    def test_1(self):
        a = np.arange(10)
        b = np.arange(10)

        res = np.zeros((a.shape[0] * b.shape[0], 2), dtype=np.int)
        cgmath.inter_index_array(a, b, res)

        ref = self._manual(a, b)

        assert_array_almost_equal(ref, res)
    


class TestIntraIndex(object):
    def _manual(self, a):
        return np.array(list(itertools.combinations(a, 2)))

    def test_1(self):
        a = np.arange(10)
        n = a.shape[0]

        res = np.zeros((n * (n-1) // 2, 2), dtype=np.int)
        cgmath.intra_index_array(a, res)

        ref = self._manual(a)

        assert_array_almost_equal(ref, res)
