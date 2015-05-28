"""Cellgrid method for coordinate analysis

Terminology:
 - box -> the entire system
 - cell -> a subset of the box
 - grid -> the division of the box into cells
 - id -> unique identifier for a cell
 - address -> the cartesian coordinates of a cell within the grid

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

def _create_views(ncells, indices, coords):
    """Create a dict relating a cell index to a view of the coords"""
    views = {}

    for i in range(ncells):
        idx = np.where(indices == i)[0]
        if len(idx) == 0:
            views[i] = coords[0:0]  # empty array
        else:
            first = idx[0]
            n = len(idx)
            views[i] = coords[first:first+n]

    return views

class CellGrid(object):
    """
    Updating the CellGrid can be done by directly setting the attributes,
    this will automatically update the CellGrid contents.

    The update method allows many attributes to be changed simultaneously
    and everything only recalculated once.
    """
    def __init__(self, box, max_dist, coordinates=None):
        """Create a grid of cells

        :Arguments:
          coordinates (n,3) - array of positions
          box (3) - size of box in each direction
          max_dist - the maximum distance to be found
        """
        self.update(box=box, max_dist=max_dist, coordinates=coordinates)

    def update(self, **kwargs):
        """Update values for this CellGrid

        Can set:
          - coordinates
          - box
          - max_dist

        This is more efficient than setting each value in turn
        as everything is only recalculated once
        """
        coords = kwargs.pop('coordinates', False)
        if not coords is False:
            self._coordinates = coords
        box = kwargs.pop('box', False)
        if not box is False:
            self._box = box
        max_dist = kwargs.pop('max_dist', False)
        if not max_dist is False:
            self._max_dist = max_dist

        # Weird logic, but arrays don't evaluate to there
        if (not box is False) or (not max_dist is False):
            self._determine_cell_dimensions()
        self._put_into_cells()

    def _determine_cell_dimensions(self):
        # number of cells in each direction
        self._ncells = np.floor_divide(self._box, self._max_dist).astype(np.int)
        self._total_cells = np.product(self._ncells)
        # size of cell in each direction
        self._cell_size = self._box / self._ncells

    def _put_into_cells(self):
        if self._coordinates is None:  # shortcut if no coords present
            return
        self._cell_addresses = self._coordinates // self._cell_size
        self._cell_indices = np.array([_address_to_id(a, self._ncells)
                                       for a in self._cell_addresses], dtype=np.int)
        self._order = self._cell_indices.argsort()
        self._sorted_coords = self._coordinates[self._order]
        self._original_indices = np.arange(len(self._coordinates))[self._order]
        self._sorted_cell_addresses = self._cell_addresses[self._order]
        self._sorted_cell_indices = self._cell_indices[self._order]
        self._views = _create_views(self._total_cells,
                                    self._sorted_cell_indices,
                                    self._sorted_coords)

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, new):
        # When setting coordinates, trigger a rebuild of the grid
        self._coordinates = new
        self._put_into_cells()

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, new):
        self._box = new
        self._determine_cell_dimensions()

    @property
    def max_dist(self):
        return self._max_dist

    @max_dist.setter
    def max_dist(self, new):
        self._max_dist = new
        self._determine_cell_dimensions()

    def __len__(self):
        return self._total_cells

    def __getitem__(self, item):
        """Retrieve a single cell"""
        try:
            view = self._views[item]
            return Cell(item, parent=self, coordinates=view)
        except KeyError:
            raise IndexError("No such item: {0}".format(item))


class Cell(object):
    """A single Cell in a CellGrid

    :Attributes:
      index
        The index of this Cell within the CellGrid
      parent
        The CellGrid to which this Cell belongs.
      coordinates
        The view on the master coordinates array on parent.

    """
    def __init__(self, idx, parent, coordinates):
        self.idx = idx
        self.parent = parent
        self.coordinates = coordinates

    @property
    def address(self):
        return _id_to_address(self.idx, self.parent._ncells)

    def __len__(self):
        return len(self.coordinates)
