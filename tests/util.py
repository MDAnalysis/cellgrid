"""Useful functions for tests"""
import numpy as np


def dist(x, y):
    v = x - y
    return np.sqrt(np.dot(v, v))


def dist_pbc(x, y, box):
    v = x - y
    v -= np.rint(v/box) * box
    return np.sqrt(np.dot(v, v))


def slow_inter_distance_nopbc(a, b):
    c = a - b[:,None]
    c = np.sqrt((c * c).sum(axis=-1))
    return c


def slow_inter_distance_withpbc(a, b, box):
    c = a - b[:, None]
    c = c - np.rint(c / box) * box
    c = np.sqrt((c * c).sum(axis=-1))
    return c


def slow_intra_distance_nopbc(a):
    n = a.shape[0]
    ref = np.zeros(n*(n-1)/2)

    pos = 0
    for i in range(n):
        for j in range(i+1, n):
            ref[pos] = dist(a[i], a[j])
            pos += 1

    return ref

def slow_intra_distance_withpbc(a, box):
    n = a.shape[0]
    ref = np.zeros(n*(n-1)/2)

    pos = 0
    for i in range(n):
        for j in range(i+1, n):
            ref[pos] = dist_pbc(a[i], a[j], box)
            pos += 1

    return ref
    
