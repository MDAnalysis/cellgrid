"""Testing for capped distance arrays


"""
import numpy as np

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
