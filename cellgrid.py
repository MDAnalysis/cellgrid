"""Cellgrid method for coordinate analysis


"""
from __future__ import division, print_function

import numpy as np

def _address_to_id(addr, ncells):
    """Return the cell index from a cell address

    address is (z, y, x)
    """
    return addr[2] + addr[1] * ncells[0] + addr[0] * ncells[1] * ncells[2]

def _id_to_address(cid, ncells):
    """Return the cell adress from a cell index

    address is (z, y, x)
    """
    z = cid // (ncells[0] * ncells[1])
    cid %= ncells[0] * ncells[1]
    y = cid // ncells[0]
    cid %= ncells[0]

    return z, y, cid

def _create_views(ncells, addresses, coords):
    """Create a dict relating a cell index to a view of the coords"""
    views = {}

    for i in range(ncells):
        idx = np.where(addresses == i)[0]
        if len(idx) == 0:
            views[i] = coords[0:0]  # empty array
        else:
            first = idx[0]
            n = len(idx)
            views[i] = coords[first:first+n]

    return views

class CellGrid(object):
    def __init__(self, box, max_dist, coordinates=None):
        """
        :Arguments:
        coordinates (n,3) array of positions
        box (3) size of box in each direction
        cellsize
        """
        self._box = box
        # Maximum desired distance
        self._max_dist = max_dist
        # determine number of cells
        self._determine_cell_dimensions()

        self._coordinates = coordinates
        if coordinates is not None:
            self._put_into_cells()

    def _determine_cell_dimensions(self):
        # number of cells in each direction
        self._ncells = np.floor_divide(self._box, self._max_dist).astype(np.int)
        self._total_cells = np.product(self._ncells)
        # size of cell in each direction
        self._cell_size = self._box / self._ncells

    def _put_into_cells(self):
        self._cell_addresses = self._coordinates // self._cell_size
        self._cell_indices = np.array([_address_to_id(a, self._ncells)
                                       for a in self._cell_addresses], dtype=np.int)
        self._order = self._cell_indices.argsort()
        self._sorted_coords = self._coordinates[self._order]
        self._original_indices = np.arange(len(self._coordinates))[self._order]
        self._sorted_cell_addresses = self._cell_addresses[self._order]
        self._sorted_cell_indices = self._cell_indices[self._order]
        self._views = _create_views(self._total_cells,
                                    self._sorted_cell_addresses,
                                    self._sorted_coords)

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, new):
        # When setting coordinates, trigger a rebuild of the grid
        self._coordinates = new
        self._put_into_cells()

    # TODO: Make changing box redo _determine_cells

    def __iter__(self):
        return iter(self.cells)


class Cell(object):
    def __init__(self):
        pass
