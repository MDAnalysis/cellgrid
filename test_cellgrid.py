import numpy as np

from cellgrid import CellGrid, _address_to_id, _id_to_address, _create_views


class TestCellDetermining(object):
    def test_creation(self):
        """Box split directly into 4 cells"""
        cg = CellGrid(box=np.ones(3), max_dist=0.25)

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([4, 4, 4]))
        assert cg._total_cells == 64

    def test_creation_2(self):
        """Box split into 4, 0.22 needs to round up to 0.25"""
        cg = CellGrid(box=np.ones(3), max_dist=0.22)

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([4, 4, 4]))
        assert cg._total_cells == 64

    def test_creation_3(self):
        cg = CellGrid(box=np.array([1.0, 1.0, 2.0]), max_dist=0.25)

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([4, 4, 8]))
        assert cg._total_cells == 128


class TestConverters(object):
    ncells = (3, 3, 3)
    nids = 27
    idx = np.arange(nids)
    addr = idx.reshape(3, 3, 3)

    def _conv(self, idx, addr):
        w = np.where(idx == addr)
        return w[0][0], w[1][0], w[2][0]

    def test_id_to_addr(self):
        addresses = (self._conv(i, self.addr) for i in range(self.nids))
    
        for i, c in zip(self.idx, addresses):
            assert _id_to_address(i, self.ncells) == c

    def test_addr_to_id(self):
        addresses = (self._conv(i, self.addr) for i in range(self.nids))

        for c, i in zip(addresses, self.idx):
            assert _address_to_id(c, self.ncells) == i

class TestCellDefinition(object):
    def test_placement(self):
        points = np.array([[0.1, 0.1, 0.1],[0.9, 0.9, 0.9]])

        cg = CellGrid(box=np.ones(3), max_dist=0.25)
        cg.coordinates = points

        assert all(cg._cell_addresses[0] == np.array([0, 0, 0]))
        assert all(cg._cell_addresses[1] == np.array([3, 3, 3]))
        assert cg._cell_indices[0] == 0
        assert cg._cell_indices[1] == 63

    def test_sorting(self):
        points = np.array([[0.8, 0.8, 0.8],
                           [0.1, 0.1, 0.1],
                           [0.9, 0.9, 0.9]])

        cg = CellGrid(box=np.ones(3), max_dist=0.25, coordinates=points)

        assert all(cg._original_indices == np.array([1, 0, 2]))
        assert all(cg._order == np.array([1, 0, 2]))
        for val, ref in zip(cg._sorted_coords, np.array([[ 0.1,  0.1,  0.1],
                                                         [ 0.8,  0.8,  0.8],
                                                         [ 0.9,  0.9,  0.9]])):
            assert all(val == ref)

        # Check that sorted coordinates are continuous C
        # important for later work
        assert cg._sorted_coords.flags['C'] == True


class TestViewCreator(object):
    # Need fancy test to make sure all content of view dict
    # are views of the original coordinate array
    # ie assert(a is view of b)
    def test_create_views(self):
        coords = np.array([0, 1, 1, 3, 4, 4])
        vals = np.arange(6) * 10

        view = _create_views(5, coords, vals)

        assert len(view) == 5
        assert len(view[0]) == 1
        assert len(view[1]) == 2
        assert len(view[2]) == 0
        assert len(view[3]) == 1
        assert len(view[4]) == 2

    def test_create_cg_views(self):
        points = np.array([[0.8, 0.8, 0.8],
                           [0.1, 0.1, 0.1],
                           [0.9, 0.9, 0.9]])

        cg = CellGrid(box=np.ones(3), max_dist=0.25, coordinates=points)

        views = cg._views
        assert len(views) == cg._total_cells
        assert all(views[0][0] == points[1])
        assert all(views[26] == points[0, 2])
