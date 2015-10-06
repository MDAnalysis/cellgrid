"""Bench distance array

Brute force (n * m) vs cellgrid method
"""
import itertools
import numpy as np

from cellgrid import CellGrid
from cellgrid.cgmath import inter_distance_array_withpbc
from cellgrid import capped_distance_array

from cgtimer import Timer

PREC = np.float32


def cg_method(a, b, box, d_max):
    cga = CellGrid(box, d_max, a)
    cgb = CellGrid(box, d_max, b)
    return capped_distance_array(cga, cgb)

def brute_force_method(a, b, box):
    res = np.zeros((a.shape[0] * b.shape[0]), dtype=PREC)
    return inter_distance_array_withpbc(a, b, box, res)

# Benchmark settings
boxsize = 10.0
d_max = 1.0

n_values = []
cg_times = []
bf_times = []
for n in itertools.chain(xrange(100, 1000, 100),
                         xrange(1000, 5000, 250)):
    a = (np.random.random(n *3).reshape(n, 3) * boxsize).astype(PREC)
    b = (np.random.random(n *3).reshape(n, 3) * boxsize).astype(PREC)
    box = (np.ones(3) * boxsize).astype(PREC)

    with Timer() as t_cg:
        cg_method(a, b, box, d_max)
    with Timer() as t_bf:
        brute_force_method(a, b, box)

    n_values.append(n)
    cg_times.append(t_cg.secs)
    bf_times.append(t_bf.secs)

    print 'CellGrid: {} Brute: {}'.format(t_cg.secs, t_bf.secs)
