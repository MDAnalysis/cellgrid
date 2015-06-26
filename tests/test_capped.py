"""Testing for capped distance arrays


"""
import numpy as np
from numpy.testing import assert_raises

from cellgrid import CellGrid
from cellgrid import capped_distance_array


class TestCappedDistanceArray(object):
    def test_capped_1(self):
        # 3 x 3 x 3 box
        box = np.ones(3) * 3
        # Equally spaced points from 0.5 to 2.5
        grid = np.array(list(np.ndindex(3, 3, 3))) + 0.5
        # A single point to use to check who it sees
        # this is in the bottom corner of the box
        probe = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box, 1.0, grid)
        cg2 = CellGrid(box, 1.0, probe)

        idx, dists = capped_distance_array(cg1, cg2)

        # Probe should have seen 27 points
        assert len(dists) == 27

    def test_capped_2(self):
        # 27 points in the corner of a
        # 10 x 10 x 10 box
        # put probe in middle and see them all
        box = np.ones(3) * 10
        # Equally spaced points from 0.5 to 2.5
        grid = np.array(list(np.ndindex(3, 3, 3))) + 0.5
        # A single point to use to check who it sees
        # this is in the bottom corner of the box
        probe = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box, 1.0, grid)
        cg2 = CellGrid(box, 1.0, probe)

        idx, dists = capped_distance_array(cg1, cg2)

        # Probe should have seen 27 points
        assert len(dists) == 27

    def test_capped_3(self):
        # 27 points in the corner of a
        # 10 x 10 x 10 box
        # put probe in middle and see only 27
        box = np.ones(3) * 10
        # Equally spaced points from 0.5 to 2.5
        grid = np.array(list(np.ndindex(10, 10, 10))) + 0.5
        # A single point to use to check who it sees
        # this is in the bottom corner of the box
        probe = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box, 1.0, grid)
        cg2 = CellGrid(box, 1.0, probe)

        idx, dists = capped_distance_array(cg1, cg2)

        # Probe should have seen 27 points
        assert len(dists) == 27

    def test_capped_periodic(self):
        box = np.ones(3) * 10

        point1 = np.array([[0.5, 1.0, 1.0]])
        point2 = np.array([[9.5, 1.0, 1.0]])

        cg1 = CellGrid(box, 1.0, point1)
        cg2 = CellGrid(box, 1.0, point2)

        idx, dists = capped_distance_array(cg1, cg2)

        assert (idx[0] == (0, 0)).all()
        assert dists[0] == 1.0

    def test_capped_partial(self):
        # 3 x 3 x 3 box
        box = np.ones(3) * 3
        # Equally spaced points from 0.5 to 2.5
        grid = np.array(list(np.ndindex(3, 3, 3))) + 0.5
        # A single point to use to check who it sees
        # this is in the bottom corner of the box
        probe = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box, 0.5, grid)
        cg2 = CellGrid(box, 0.5, probe)

        idx, dists = capped_distance_array(cg1, cg2)

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

        assert_raises(ValueError, capped_distance_array, cg1, cg2)

    def test_compatibility_2(self):
        # Same box volume, will have differing number of cells in
        # each dimension
        box1 = np.array([2.0, 3.0, 4.0])
        box2 = np.array([4.0, 2.0, 3.0])

        point1 = np.array([[0.5, 0.5, 0.5]])
        point2 = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box1, 0.5, point1)
        cg2 = CellGrid(box2, 0.5, point2)

        assert_raises(ValueError, capped_distance_array, cg1, cg2)

    def test_compatibility_3(self):
        # Same box size, different cutoff
        box1 = np.ones(3) * 5.0
        box2 = np.ones(3) * 5.0

        point1 = np.array([[0.5, 0.5, 0.5]])
        point2 = np.array([[1.0, 1.0, 1.0]])

        cg1 = CellGrid(box1, 0.5, point1)
        cg2 = CellGrid(box2, 0.25, point2)

        assert_raises(ValueError, capped_distance_array, cg1, cg2)
