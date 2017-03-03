import numpy as np
from numpy.testing import assert_array_almost_equal
import itertools

from cellgrid import cgmath

import util

class TestInterDistanceNoPBC(object):
    precs = [np.float32, np.float64]
    def test_1(self):
        for prec in self.precs:
            a = np.arange(30).reshape(10, 3).astype(prec)
            b = np.arange(30).reshape(10, 3).astype(prec)
            res = np.zeros(100).astype(prec)

            cgmath.inter_distance_array_nopbc(a, b, res)
            ref = np.ravel(util.slow_inter_distance_nopbc(a, b))

            assert_array_almost_equal(ref, res)


class TestInterDistanceWithPBC(object):
    precs = [np.float32, np.float64]
    def test_1(self):
        for prec in self.precs:
            a = np.arange(30).reshape(10, 3).astype(prec)
            b = np.arange(30).reshape(10, 3).astype(prec)
            res = np.zeros(100).astype(prec)
            box = np.ones(3).astype(prec) * 30

            cgmath.inter_distance_array_withpbc(a, b, box, res)
            ref = np.ravel(util.slow_inter_distance_withpbc(a, b, box))

            assert_array_almost_equal(ref, res)


class TestIntraDistanceNoPBC(object):
    precs = [np.float32, np.float64]
    def test_1(self):
        for prec in self.precs:
            a = np.arange(30).reshape(10, 3).astype(prec)
            n = 10
            res = np.zeros(n * (n-1) // 2, dtype=prec)

            cgmath.intra_distance_array_nopbc(a, res)
            ref = util.slow_intra_distance_nopbc(a)

            assert_array_almost_equal(ref, res)

class TestIntraDistanceWithPBC(object):
    precs = [np.float32, np.float64]
    def test_1(self):
        for prec in self.precs:
            a = np.arange(30).reshape(10, 3).astype(prec)
            n = 10
            res = np.zeros(n * (n-1) // 2, dtype=prec)
            box = np.ones(3).astype(prec) * 30

            cgmath.intra_distance_array_withpbc(a, box, res)
            ref = util.slow_intra_distance_withpbc(a, box)

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
