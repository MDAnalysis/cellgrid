import cProfile
import numpy as np

import cellgrid

box = np.ones(3) * 10

n = 10000

coords = np.random.random(n * 3).reshape(n, 3)

