"""Functions which rely upon CellGrids to do their calculations

From coordinates:

  capped_distance_array(a, b, max_dist, box)

  capped_self_distance_array(a, max_dist, box)

From cellgrids:

  cellgrid_distance_array(cga, cgb)

  cellgrid_self_distance_array(cga)

"""
import numpy as np
try:
    from itertools import izip
except ImportError:
    izip = zip
import itertools

from cellgrid import cgmath
from cellgrid.core import CellGrid


def capped_distance_array(a, b, cellsize, box=None):
    """Calculate all pairwise distances between *a* and *b* up to *max_dist*

    If given, *box* defines the unit cell size for periodic boundaries
    """
    cga = CellGrid(box, cellsize, a)
    cgb = CellGrid(box, cellsize, b)

    return cellgrid_distance_array(cga, cgb)




def cellgrid_distance_array(cg1, cg2):
    """Calculate all pairwise distances between pairs in cg1 and cg2

    :Returns:
      indices, distances

    indices - (n, 2) array of indices
              The first index refers to coordinates within cg1
              The second index from cg2
    distances - (n) Array of distances
    """
    if not cg1 == cg2:
        raise ValueError("CellGrids are not compatible")

    # calculate required size of array
    Nreq = _calculate_distance_array_size(cg1, cg2)
    dist = np.empty(Nreq, dtype=cg1.datatype)
    indices = np.empty((Nreq, 2), dtype=np.int)

    box = cg1.box

    # Calculate distances cell by cell
    pos = 0
    # For each cell in cg1
    for cell in cg1:
        # Iterate over all neighbours in other cellgrid
        for addr in itertools.chain([cell.address], cell.all_neighbours):
            other = cg2[addr]
            cgmath.inter_distance_array_withpbc(
                cell.coordinates,
                other.coordinates,
                box,
                dist[pos:]
            )
            cgmath.inter_index_array(
                cell.indices,
                other.indices,
                indices[pos:]
            )
            pos += len(cell) * len(other)

    return indices, dist


def _calculate_distance_array_size(cg1, cg2):
    """Calculate the required size for results"""
    N = 0
    for cell in cg1:
        na = len(cell)
        for neb_addr in itertools.chain([cell.address], cell.all_neighbours):
            # Find corresponding cell in other cg
            other = cg2[neb_addr]
            N += na * len(other)
    return N


def capped_self_distance_array(a, cellsize, box, particles_per_cell = 30):
    """
    Optimized to identify pair contacts within a distance
    Cellsize is changed based on the maximum of number of 
    particles per cell and provided cellsize
    """
    if particles_per_cell is not None:
        cellsize = max(cellsize, _per_cell(a.shape[0], box, particles_per_cell))
    cga = CellGrid(box, cellsize, a)

    return cellgrid_self_distance_array(cga)

def _per_cell( N, box, particles_per_cell):
    """
    Returns the cell size based on the optimum particles per cell
    """
    return np.cbrt((np.product(box) * particles_per_cell) / N)

def cellgrid_self_distance_array(cg1):
    """Calculate all pairwise distances within a certain CellGrid

    :Returns:
     indices, distances

    indices - (n, 2)
    distances (n)
    """
    box = cg1.box

    Nreq = _calculate_self_distance_array_size(cg1)
    indices = np.empty((Nreq, 2), dtype=np.int)
    dist = np.empty(Nreq, dtype=cg1.datatype)

    pos = 0
    for cell in cg1:
        n = len(cell)
        if n > 1:
            # Do own cell as a self distance comparison
            cgmath.intra_distance_array_withpbc(
                cell.coordinates,
                box,
                dist[pos:]
            )
            cgmath.intra_index_array(
                cell.indices,
                indices[pos:]
            )
            pos += n * (n - 1) // 2
        # Then all half neighbours as a full comparison
        for addr in cell.half_neighbours:
            other = cg1[addr]
            if not other:
                continue
            cgmath.inter_distance_array_withpbc(
                cell.coordinates,
                other.coordinates,
                box,
                dist[pos:]
            )
            cgmath.inter_index_array(
                cell.indices,
                other.indices,
                indices[pos:]
            )
            pos += n * len(other)

    return indices, dist

def _calculate_self_distance_array_size(cg):
    """Pass back an index and coordinate array of correct size
    for a capped self distance array
    """
    # N is the total number of comparisons we will make
    N = 0
    for c in cg:
        # number in this cell
        nc = len(c)
        # can make 1/2 n * (n-1) inside the same cell
        N += (nc - 1) * nc // 2
        for addr in c.half_neighbours:
            o = cg[addr]
            N += nc * len(o)
    return N
