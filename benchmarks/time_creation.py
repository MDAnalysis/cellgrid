"""Benchmark creating CellGrids"""
import timeit
import numpy as np

import cellgrid

stp = """\
import cellgrid
import numpy as np

coords = np.random.random({n} * 3).reshape({n}, 3) * 10
box = np.ones(3) * 10
"""

for n in [100, 1000, 10000, 100000]:
    val = timeit.timeit('cg = cellgrid.CellGrid(box, 2, coords)',
                        setup=stp.format(n=n), number=100)
    print n, val
