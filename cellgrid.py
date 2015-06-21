"""Cellgrid method for coordinate analysis

Terminology:
 - box -> the entire extent system
 - cell -> a subset of the box, a box contains many cells
 - grid -> the division of the box into equal sized cells
 - index -> unique identifier for a cell as an integer
 - address -> the cartesian coordinates of a cell within the grid
              (x, y, z), also uniquely identifies cells


This is an implementation of a cell method as described in 
Allen and Tildesley page 149.  Rather than using a linked list
for each cell, coordinates are instead sorted according to their
cells.  This then allows cells to be defined as a continuous slice
of a numpy array (essentially a pointer and size).  This approach
was chosen so that interfacing with external C code was simpler.



"""
from __future__ import division, print_function

import numpy as np
from itertools import izip

def _address_to_index(addr, ncells):
    """Return the cell index from a cell address

    address is (x, y, z)
    """
    return addr[0] + addr[1] * ncells[0] + addr[2] * ncells[1] * ncells[0]

def _index_to_address(cid, ncells):
    """Return the cell adress from a cell index

    address is (x, y, z)
    """
    z = cid // (ncells[0] * ncells[1])
    cid %= ncells[0] * ncells[1]
    y = cid // ncells[0]
    cid %= ncells[0]

    return cid, y, z

def _create_views(ncells, indices, coords, original):
    """Create a list relating a cell index to a view of the coords

    Each entry is a tuple of two pointers, one to coordinates and
    the other to indices
    """
    views = []

    for i in range(ncells):
        idx = np.where(indices == i)[0]
        if len(idx) == 0:
            views.append((coords[0:0], original[0:0]))  # empty array
        else:
            first = idx[0]
            n = len(idx)
            views.append((coords[first:first+n], original[first:first+n]))

    return views

# TODO: Move distance stuff out into a separate file
# This file should only be for Cells and CellGrids
def dist(x, y):
    """Distance between two points"""
    return np.linalg.norm(x - y)

def intra_distance_array(coords, indices,
                         out_d, out_idx,
                         offset):
    """Calculate all pairs within a set of coords"""
    pos = 0
    for i, (ac, ai) in enumerate(izip(coords, indices)):
        for bc, bi in izip(coords[i+1:], indices[i+1:]):
            out_idx[offset + pos] = ai, bi
            out_d[offset + pos] = dist(ac, bc)
            pos += 1

def inter_distance_array(coords1, indices1,
                         coords2, indices2,
                         out_d, out_idx,
                         offset):
    """Calculate all pairs within two sets of coords"""
    pos = 0
    for ac, ai in izip(coords1, indices1):
        for bc, bi in izip(coords2, indices2):
            out_idx[offset + pos] = ai, bi
            out_d[offset + pos] = dist(ac, bc)
            pos += 1


class CellGrid(object):
    """
    Updating the CellGrid can be done by directly setting the attributes,
    this will automatically update the CellGrid contents.

    The update method allows many attributes to be changed simultaneously
    and everything only recalculated once.

    Iterating over the CellGrid will cycle through all the Cells
    """
    def __init__(self, box, max_dist, coordinates=None):
        """Create a grid of cells

        :Arguments:
          box (3) - size of box in each direction
          max_dist - the maximum distance to be found
          coordinates (n,3) - array of positions [optional]
        """
        self.update(box=box, max_dist=max_dist, coordinates=coordinates)
        self.periodic = True

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
        """Process the coordinates into cells"""
        # Can probably remove a lot of these arrays,
        # as they are unused
        # But waiting to do optimisation
        if self._coordinates is None:  # shortcut if no coords present
            return
        # Which cell each coordinate is in (as address)
        self._cell_addresses = self._coordinates // self._cell_size
        # Which cell each coordinate is in (as index)
        self._cell_indices = np.array([self._address_to_index(a)
                                       for a in self._cell_addresses], dtype=np.int)
        # An array that puts everything into correct order
        self._order = self._cell_indices.argsort()
        # Coordinates sorted according to their cell
        self._sorted_coords = self._coordinates[self._order]
        # The original ordering of the coordinates
        # can use to "unsort" coordinates
        self._original_indices = np.arange(len(self._coordinates))[self._order]
        self._sorted_cell_addresses = self._cell_addresses[self._order]
        # A sorted version of the cell indices
        # will be "blocks" of cell indices
        # eg [0, 0, 0, 1, 1, 2, 2, 2, ....]
        # allowing the slices to be created
        self._sorted_cell_indices = self._cell_indices[self._order]
        self._views = _create_views(self._total_cells,
                                    self._sorted_cell_indices,
                                    self._sorted_coords,
                                    self._original_indices)

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

    def _address_to_index(self, address):
        """Translate an address inside this CellGrid to an index"""
        return _address_to_index(address, self._ncells)

    def _index_to_address(self, index):
        """Translate an index to an address inside this CellGrid"""
        return _index_to_address(index, self._ncells)

    def _address_pbc(self, address):
        """Apply periodic boundary conditions to an address"""
        w = address >= self._ncells
        address[w] -= self._ncells[w]

        w = address < 0
        address[w] += self._ncells[w]

        return address

    # TODO: Move this out of the class
    # and into its own submodule
    def capped_distance_array(self):
        """Return the capped distance array

        This is given as 2 arrays, one of indices with shape (N, 2)
        and another of distances with shape (N)

        Where N is the number of pairs that the method generates
        """
        idx, dists = self._allocate_results()
        # Iterate through and fill outputs
        pos = 0
        for c in self:  # loop over all cells
            print("Doing cell {}".format(c))
            nc = len(c)
            size = (nc - 1) * nc // 2
            print("{} {}".format(pos, pos+size))
            intra_distance_array(c.coordinates, c.indices,
                                 dists, idx,
                                 pos)
            pos += size
            for neb in c.neighbours:  # loop over 13 nebs
                size = nc * len(neb)
                inter_distance_array(c.coordinates, c.indices,
                                     neb.coordinates, neb.indices,
                                     dists, idx,
                                     pos)
                pos += size

        return idx, dists

    def _allocate_results(self):
        """Pass back an index and coordinate array of correct size"""
        # This is in the wrong place, should move into
        # capped_self_distance_array once created
        N = 0
        for c in self:
            nc = len(c)
            N += (nc - 1) * nc // 2
            for o in c.neighbours:
                N += nc * len(o)
        # Allocate outputs
        idx = np.zeros((N, 2), dtype=np.int)
        dists = np.zeros(N)

        return idx, dists

    def __getitem__(self, item):
        """Retrieve a single cell
        
        Can use either an address (as np array or tuple) or integer index
        """
        # Internally, cells are stored in a dict of views
        if isinstance(item, tuple):
            item = np.array(item)
        if isinstance(item, np.ndarray):  # if address
            if self.periodic:
                item = self._address_pbc(item)
            item = self._address_to_index(item)

        try:
            coords, idx = self._views[item]
        except KeyError:
            raise IndexError("No such item: {0}".format(item))
        return Cell(item, parent=self, coordinates=coords, indices=idx)

    def __eq__(self, other):
        """Check that this CellGrid is compatible with another
        Checks:
         - box size
         - number of cells in each dimension

        This is important when wanting to compare the contents of the two
        Note that the two CellGrids may have a different coordinate data, both
        in terms of data and number of coordinates,
        and still be considered equal.
        """
        if not isinstance(other, CellGrid):
            return False
        if not len(other) == len(self):
            return False
        if not (other._ncells == self._ncells).all():
            return False

        return True

    def __repr__(self):
        return ("<CellGrid with dimensions {nc[0]}, {nc[1]}, {nc[2]}>"
                "".format(nc=self._ncells))


class Cell(object):
    """A single Cell in a CellGrid

    :Attributes:
      index
        The index of this Cell within the CellGrid
      address
        The address of this Cell within the CellGrid
      parent
        The CellGrid to which this Cell belongs.
      coordinates
        The view on the master coordinates array on parent.
      original
        The indices of the coordinates
    """
    _half_route = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
                            [1, 0, -1], [1, 1, -1], [0, 1, -1], [-1, 1, -1],
                            [1, 0, 1], [1, 1, 1], [0, 1, 1], [-1, 1, 1], [0, 0, 1]])

    _full_route = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
                            [1, 0, -1], [1, 1, -1], [0, 1, -1], [-1, 1, -1],
                            [1, 0, 1], [1, 1, 1], [0, 1, 1], [-1, 1, 1], [0, 0, 1],
                            [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                            [-1, 0, -1], [-1, -1, -1], [0, -1, -1], [1, -1, -1], [0, 0, -1],
                            [-1, 0, 1], [-1, -1, 1], [0, -1, 1], [1, -1, 1]])


    # Cells are generated on demand by CellGrids, so they
    # should ideally be as light as possible
    def __init__(self, index, parent, coordinates, indices):
        self.index = index
        self.parent = parent
        self.coordinates = coordinates
        self.indices = indices

    def __len__(self):
        return len(self.coordinates)

    @property
    def address(self):
        """The cartesian address of this Cell within its CellGrid"""
        return _index_to_address(self.index, self.parent._ncells)

    # This method of generating neighbours only allows me to
    # retrieve cells from within the same CellGrid, which is
    # annoying for doing 2 species comparison
    #
    # Will want to change this to return either indices or 
    # addresses of neighbours. 
    # neighbours_indices
    # neighbours_coordinates
    #
    # The periodicity will still be determined by the CellGrid
    # however!
    # Should probably move knowledge of boundaries into the Cell,
    # ie. the cell is aware of the PBC.
    @property
    def half_neighbours(self):
        """Generator to iterate over the address of my 13 neighbours"""
        me = self.address
        return (me + other for other in self._half_route)

    @property
    def all_neighbours(self):
        """Generator to iterate over the address of my 26 neighbours"""
        me = self.address
        return (me + other for other in self._full_route)

    def __repr__(self):
        return ("<Cell at {addr} with {num} coords>"
                "".format(addr=self.address, num=len(self)))
