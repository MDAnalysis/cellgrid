"""Fancy maths functions

Currently all for floats/real*4.
Would be nice to have some variables at top define precision,
then we could compile twice with different precision each time.

distance_arrays
Calculate distance between points

Comes in two flavours:
 - nopbc (no periodic boundaries)
 - withpbc (with periodic boundaries, penultimate argument is box data)

index_arrays
Combine two arrays of indices into a (n*m, 2)

inter
Given two arrays, do all pairwise comparisons (n*m)

intra
Given a single array, do all unique pairwise comparisons 1/2n*(n-1)
"""

import numpy as np
cimport cython
cimport numpy as np

from libc.math cimport (
    sqrt,
    fabs,
)

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def inter_distance_array_nopbc(np.ndarray[np.float32_t, ndim=2] coords1,
                               np.ndarray[np.float32_t, ndim=2] coords2,
                               np.ndarray[np.float32_t, ndim=1] results):
    cdef Py_ssize_t i, j, k, pos
    cdef DTYPE_t rsq, dx[3], rij[3]

    pos = 0

    for i in range(coords1.shape[0]):
        for k in range(3):
            rij[k] = coords1[i, k]

        for j in range(coords2.shape[0]):
            for k in range(3):
                dx[k] = rij[k] - coords2[j, k]
            rsq = 0.0
            for k in range(3):
                rsq += dx[k] * dx[k]
            results[pos] = sqrt(rsq)
            pos += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def inter_distance_array_withpbc(np.ndarray[np.float32_t, ndim=2] coords1,
                                 np.ndarray[np.float32_t, ndim=2] coords2,
                                 np.ndarray[np.float32_t, ndim=1] box,
                                 np.ndarray[np.float32_t, ndim=1] results):

    cdef Py_ssize_t i, j, k, pos
    cdef DTYPE_t rsq, dx[3], bhalf[3]

    for k in range(3):
        bhalf[k] = box[k] / 2.0

    pos = 0

    for i in range(coords1.shape[0]):
        for j in range(coords2.shape[0]):
            for k in range(3):
                dx[k] = coords1[i, k] - coords2[j, k]
            # Periodic boundaries
            for k in range(3):
                if fabs(dx[k]) > bhalf[k]:
                    if dx[k] < 0:
                        dx[k] += box[k]
                    else:
                        dx[k] -= box[k]
            rsq = 0.0
            for k in range(3):
                rsq += dx[k] * dx[k]
            results[pos] = sqrt(rsq)
            pos += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def intra_distance_array_nopbc(np.ndarray[np.float32_t, ndim=2] coords1,
                               np.ndarray[np.float32_t, ndim=1] results):
    cdef Py_ssize_t i, j, k, pos
    cdef DTYPE_t rsq, dx[3], rij[3]

    pos = 0

    for i in range(coords1.shape[0]):
        for k in range(3):
            rij[k] = coords1[i, k]

        for j in range(i + 1, coords1.shape[0]):
            for k in range(3):
                dx[k] = rij[k] - coords1[j, k]
            rsq = 0.0
            for k in range(3):
                rsq += dx[k] * dx[k]
            results[pos] = sqrt(rsq)
            pos += 1
            

@cython.boundscheck(False)
@cython.wraparound(False)
def intra_distance_array_withpbc(np.ndarray[np.float32_t, ndim=2] coords1,
                                 np.ndarray[np.float32_t, ndim=1] box,
                                 np.ndarray[np.float32_t, ndim=1] results):
    cdef Py_ssize_t i, j, k, pos
    cdef DTYPE_t rsq, dx[3], bhalf[3]

    for k in range(3):
        bhalf[k] = box[k] / 2.0

    pos = 0

    for i in range(coords1.shape[0]):
        for j in range(i + 1, coords1.shape[0]):
            for k in range(3):
                dx[k] = coords1[i, k] - coords1[j, k]
            # Periodic boundaries
            for k in range(3):
                if fabs(dx[k]) > bhalf[k]:
                    if dx[k] < 0:
                        dx[k] += box[k]
                    else:
                        dx[k] -= box[k]
            rsq = 0.0
            for k in range(3):
                rsq += dx[k] * dx[k]
            results[pos] = sqrt(rsq)
            pos += 1
            

@cython.boundscheck(False)
@cython.wraparound(False)
def inter_index_array(np.ndarray[np.int64_t, ndim=1] idx1,
                      np.ndarray[np.int64_t, ndim=1] idx2,
                      np.ndarray[np.int64_t, ndim=2] out):
    cdef Py_ssize_t i, j, pos

    pos = 0

    for i in range(idx1.shape[0]):
        for j in range(idx2.shape[0]):
            out[pos, 0] = idx1[i]
            out[pos, 1] = idx2[j]
            pos += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def intra_index_array(np.ndarray[np.int64_t, ndim=1] idx1,
                      np.ndarray[np.int64_t, ndim=2] out):
    cdef Py_ssize_t i, j, pos

    pos = 0

    for i in range(idx1.shape[0]):
        for j in range(i + 1, idx1.shape[0]):
            out[pos, 0] = idx1[i]
            out[pos, 1] = idx1[j]
            pos += 1
