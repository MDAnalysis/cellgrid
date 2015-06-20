import numpy as np

from cellgrid import CellGrid, _address_to_index, _index_to_address, _create_views


class TestCellDetermining(object):
    def test_creation(self):
        """Box split directly into 4 cells"""
        cg = CellGrid(box=np.ones(3), max_dist=0.25)

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([4, 4, 4]))
        assert cg._total_cells == 64
        assert len(cg) == 64

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

    def test_change_box(self):
        """Redefine box"""
        cg = CellGrid(box=np.ones(3), max_dist=0.25)

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([4, 4, 4]))
        assert cg._total_cells == 64
        assert len(cg) == 64

        cg.box = np.ones(3) * 2

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([8, 8, 8]))
        assert cg._total_cells == 512
        assert len(cg) == 512

    def test_change_max_dist(self):
        """Redefine max dist"""
        cg = CellGrid(box=np.ones(3), max_dist=0.25)

        assert all(cg._cell_size == np.array([0.25, 0.25, 0.25]))
        assert all(cg._ncells == np.array([4, 4, 4]))
        assert cg._total_cells == 64
        assert len(cg) == 64

        cg.max_dist = 0.50

        assert all(cg._cell_size == np.array([0.50, 0.50, 0.50]))
        assert all(cg._ncells == np.array([2, 2, 2]))
        assert cg._total_cells == 8
        assert len(cg) == 8


class TestConverters(object):
    ncells = (3, 3, 3)
    nids = 27
    idx = np.arange(nids)
    addr = idx.reshape(3, 3, 3)

    def _conv(self, idx, addr):
        w = np.where(idx == addr)
        return w[2][0], w[1][0], w[0][0]

    def test_index_to_addr(self):
        addresses = (self._conv(i, self.addr) for i in range(self.nids))
    
        for i, c in zip(self.idx, addresses):
            assert _index_to_address(i, self.ncells) == c

    def test_addr_to_index(self):
        addresses = (self._conv(i, self.addr) for i in range(self.nids))

        for c, i in zip(addresses, self.idx):
            assert _address_to_index(c, self.ncells) == i


class TestCellDefinition(object):
    def test_placement(self):
        points = np.array([[0.1, 0.1, 0.1],
                           [0.9, 0.9, 0.9]])

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

    def test_nocoords(self):
        cg = CellGrid(box=np.ones(3), max_dist=0.25)
        cg.coordinates = None

        assert cg.coordinates is None


class TestViewCreator(object):
    # Need fancy test to make sure all content of view dict
    # are views of the original coordinate array
    # ie assert(a is view of b)
    def test_create_views(self):
        coords = np.array([0, 1, 1, 3, 4, 4])
        indices = np.array([0, 1, 2, 3, 4, 5])
        vals = np.arange(6) * 10

        view = _create_views(5, coords, vals, indices)

        assert len(view) == 5
        assert len(view[0][0]) == 1
        assert len(view[1][0]) == 2
        assert len(view[2][0]) == 0
        assert len(view[3][0]) == 1
        assert len(view[4][0]) == 2

    def test_create_cg_views(self):
        points = np.array([[0.8, 0.8, 0.8],
                           [0.1, 0.1, 0.1],
                           [0.9, 0.9, 0.9]])

        cg = CellGrid(box=np.ones(3), max_dist=0.25, coordinates=points)

        views = cg._views
        assert len(views) == cg._total_cells
        assert all(views[0][0][0] == points[1])
        assert all(views[26][0] == points[0, 2])

    def test_create_cg_views_2(self):
        # 8 cells
        # 1 coordinate in cell 0 (first)
        # 2 coordinates in cell 7 (last)
        points = np.array([[0.8, 0.8, 0.8],
                           [0.1, 0.1, 0.1],
                           [0.9, 0.9, 0.9]])

        cg = CellGrid(box=np.ones(3), max_dist=0.50,
                      coordinates=points)

        for cell, exp in zip(cg, [1, 0, 0, 0,
                                  0, 0, 0, 2]):
            assert len(cell) == exp
