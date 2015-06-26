"""Functions which rely upon CellGrids to do their calculations

capped_distance_array(cg1, cg2)

capped_self_distance_array(cg1)

"""
import numpy as np
try:
    from itertools import izip
except ImportError:
    izip = zip


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
                         box,
                         out_d, out_idx,
                         offset):
    """Calculate all pairs within two sets of coords"""
    pos = 0
    for ac, ai in izip(coords1, indices1):
        for bc, bi in izip(coords2, indices2):
            out_idx[offset + pos] = ai, bi
            out_d[offset + pos] = dist(ac, bc, box)
            pos += 1


def dist(x, y, b):
    """Distance between two points with periodic boundaries"""
    dx = y - x
    dx -= np.rint(dx / b) * b
    return np.sqrt((dx * dx).sum())


def capped_distance_array(cg1, cg2, result=None):
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
    if result is None:
        dist = np.empty(Nreq, dtype=np.float32)
        indices = np.empty((Nreq, 2), dtype=np.int)
    # TODO: check size of passed array if passed one

    box = cg1.box

    # Calculate distances cell by cell
    pos = 0
    for cell in cg1:
        other = cg2[cell.address]
        inter_distance_array(cell.coordinates, cell.indices,
                             other.coordinates, other.indices,
                             box,
                             dist, indices,
                             pos)
        pos += len(cell) * len(other)
        for other_addr in other.all_neighbours:
            other = cg2[other_addr]
            inter_distance_array(cell.coordinates, cell.indices,
                                 other.coordinates, other.indices,
                                 box,
                                 dist, indices,
                                 pos)
            pos += len(cell) * len(other)

    return indices, dist


def _calculate_distance_array_size(cg1, cg2):
    """Calculate the required size for results"""
    N = 0
    for cell in cg1:
        na = len(cell)
        addr = cell.address
        # Find corresponding cell in other cg
        other = cg2[addr]
        N += na * len(other)
        # Loop over all 26 neighbours to this cell
        for neb_addr in cell.all_neighbours:
            # And retreve that cell in the other cg
            other = cg2[neb_addr]
            N += na * len(other)
    return N


def capped_self_distance_array(cg1, result=None):
    """Calculate all pairwise distances within a certain CellGrid

    :Returns:
     indices, distances

    indices - (n, 2)
    distances (n)
    """
    pass


def _allocate_self_results(cg):
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
        for o in c.half_neighbours:
            N += nc * len(o)

    # Allocate outputs
    idx = np.zeros((N, 2), dtype=np.int)
    dists = np.zeros(N)

    return idx, dists
