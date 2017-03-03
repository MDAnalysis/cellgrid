"""Testing for capped distance arrays


"""
import numpy as np
from numpy.testing import assert_raises

from cellgrid import CellGrid
from cellgrid import cgmath
from cellgrid import (
    cellgrid_distance_array,
    capped_distance_array,
    cellgrid_self_distance_array,
    capped_self_distance_array,
)


class TestCappedDistanceArray(object):
    precs = [np.float32, np.float64]
    def test_capped_1(self):
        for prec in self.precs:
            # 3 x 3 x 3 box
            box = np.ones(3).astype(prec) * 3
            # Equally spaced points from 0.5 to 2.5
            grid = np.array(list(np.ndindex(3, 3, 3))).astype(prec) + 0.5
            # A single point to use to check who it sees
            # this is in the bottom corner of the box
            probe = np.array([[1.0, 1.0, 1.0]]).astype(prec)

            cg1 = CellGrid(box, 1.0, grid)
            cg2 = CellGrid(box, 1.0, probe)

            idx, dists = cellgrid_distance_array(cg1, cg2)

            # Probe should have seen 27 points
            assert len(dists) == 27

    def test_capped_2(self):
        for prec in self.precs:
            # 27 points in the corner of a
            # 10 x 10 x 10 box
            # put probe in middle and see them all
            box = np.ones(3).astype(prec) * 10
            # Equally spaced points from 0.5 to 2.5
            grid = np.array(list(np.ndindex(3, 3, 3))).astype(prec) + 0.5
            # A single point to use to check who it sees
            # this is in the bottom corner of the box
            probe = np.array([[1.0, 1.0, 1.0]]).astype(prec)

            cg1 = CellGrid(box, 1.0, grid)
            cg2 = CellGrid(box, 1.0, probe)

            idx, dists = cellgrid_distance_array(cg1, cg2)

        # Probe should have seen 27 points
            assert len(dists) == 27

    def test_capped_3(self):
        for prec in self.precs:
            # 27 points in the corner of a
            # 10 x 10 x 10 box
            # put probe in middle and see only 27
            box = (np.ones(3) * 10).astype(prec)
            # Equally spaced points from 0.5 to 2.5
            grid = (np.array(list(np.ndindex(10, 10, 10))) + 0.5).astype(prec)
            # A single point to use to check who it sees
            # this is in the bottom corner of the box
            probe = (np.array([[1.0, 1.0, 1.0]])).astype(prec)

            cg1 = CellGrid(box, 1.0, grid)
            cg2 = CellGrid(box, 1.0, probe)

            idx, dists = cellgrid_distance_array(cg1, cg2)

        # Probe should have seen 27 points
            assert len(dists) == 27

    def test_capped_periodic(self):
        for prec in self.precs:
            box = (np.ones(3) * 10).astype(prec)

            point1 = np.array([[0.5, 1.0, 1.0]]).astype(prec)
            point2 = np.array([[9.5, 1.0, 1.0]]).astype(prec)

            cg1 = CellGrid(box, 1.0, point1)
            cg2 = CellGrid(box, 1.0, point2)

            idx, dists = cellgrid_distance_array(cg1, cg2)

            assert (idx[0] == (0, 0)).all()
            assert dists[0] == 1.0

    def test_capped_partial(self):
        for prec in self.precs:
            # 3 x 3 x 3 box
            box = (np.ones(3) * 3).astype(prec)
            # Equally spaced points from 0.5 to 2.5
            grid = (np.array(list(np.ndindex(3, 3, 3))) + 0.5).astype(prec)
            # A single point to use to check who it sees
            # this is in the bottom corner of the box
            probe = np.array([[1.0, 1.0, 1.0]]).astype(prec)

            cg1 = CellGrid(box, 0.5, grid)
            cg2 = CellGrid(box, 0.5, probe)

            idx, dists = cellgrid_distance_array(cg1, cg2)

            # Probe should have seen 8 points
            assert len(dists) == 8

    def test_compatibility_1(self):

        # Different box sizes, should raise ValueError
        box1 = np.ones(3)
        box2 = np.ones(3) * 5.0

        point1 = np.array([[0.5, 0.5, 0.5]])
        point2 = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box1, 0.5, point1)
        cg2 = CellGrid(box2, 0.5, point2)

        assert_raises(ValueError, cellgrid_distance_array, cg1, cg2)

    def test_compatibility_2(self):
        # Same box volume, will have differing number of cells in
        # each dimension
        box1 = np.array([2.0, 3.0, 4.0])
        box2 = np.array([4.0, 2.0, 3.0])

        point1 = np.array([[0.5, 0.5, 0.5]])
        point2 = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box1, 0.5, point1)
        cg2 = CellGrid(box2, 0.5, point2)

        assert_raises(ValueError, cellgrid_distance_array, cg1, cg2)

    def test_compatibility_3(self):
        # Same box size, different cutoff
        box1 = np.ones(3) * 5.0
        box2 = np.ones(3) * 5.0

        point1 = np.array([[0.5, 0.5, 0.5]])
        point2 = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box1, 0.5, point1)
        cg2 = CellGrid(box2, 0.25, point2)

        assert_raises(ValueError, cellgrid_distance_array, cg1, cg2)


class TestRandomCappedWithPBC(object):
    precs = [np.float32,np.float64]

    def test_1(self):
        for prec in self.precs:
        # 2 sets of 100 points in 10*10*10 box
            n = 100
            boxsize = 10.0
            d_max = 2.0

            a = (np.random.random(n * 3).reshape(n, 3) * boxsize).astype(prec)
            b = (np.random.random(n * 3).reshape(n, 3) * boxsize).astype(prec)
            box = (np.ones(3) * boxsize).astype(prec)

            idx, dists = capped_distance_array(a, b, d_max, box)

            # Brute force approach, do all n * n comparisons
            ref = np.zeros(n * n, dtype=prec)
            cgmath.inter_distance_array_withpbc(a, b, box, ref)
            ref = ref.reshape(n, n)

            # Check all distances under 2.0 were caught
            ref_idx = np.where(ref < d_max)
            for x, y in zip(ref_idx[0], ref_idx[1]):
                # Order can be reversed
                assert (x, y in idx) or (y, x in idx)
            # Check all reported distances were accurate
            # ie, ref[idx[i]] == dists[i]
            for (i, j), d in zip(idx, dists):
                assert ref[i, j] == d


class TestRandomSelfArray(object):
    precs = [np.float32,np.float64]
    def test_1(self):
        for prec in self.precs:
            n = 100
            boxsize = 10.0
            d_max = 2.0

            a = (np.random.random(n * 3).reshape(n, 3) * boxsize).astype(prec)
            box = (np.ones(3) * boxsize).astype(prec)

            idx, dists = capped_self_distance_array(a, d_max, box)

            ref = np.zeros(n * n, dtype=prec)
            cgmath.inter_distance_array_withpbc(a, a, box, ref)
            ref = ref.reshape(n, n)
            # Mask out coordinates seeing themselves
            ref[np.diag_indices_from(ref)] = d_max + 1.0

            ref_idx = np.where(ref < d_max)
            for x, y in zip(ref_idx[0], ref_idx[1]):
                assert (x, y in idx) or (y, x in idx)

            for (i, j), d in zip(idx, dists):
                assert ref[i, j] == d
