import numpy as np

from cellgrid import CellGrid, _address_to_id, _id_to_address


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
