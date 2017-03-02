CellGrid
========

[![Build Status](https://img.shields.io/travis/MDAnalysis/cellgrid.svg)](https://travis-ci.org/MDAnalysis/cellgrid)
[![Coverage](https://img.shields.io/coveralls/MDAnalysis/cellgrid.svg)](https://coveralls.io/r/MDAnalysis/cellgrid?branch=master)

Cellgrid offers scalable functions for calculating pairwise distances between arrays of 3d coordinates.
For many cases the distances only up to a given value are of interest, meaning that the volume can be decomposed into smaller subvolumes to reduce the number of pairs that need to be calculated.
CellGrid was designed with molecular dynamics results in mind, and offers support for periodic boundary conditions.

Install me like this:
---------------------
``` bash
git clone git@github.com:MDAnalysis/cellgrid.git
cd cellgrid
python setup.py install --user
```

How to use me
-------------

``` python
>>> import numpy as np
>>> from cellgrid import capped_distance_array

# Random coordinates to search for pairs between
>>> a = np.random.random(3000).reshape(1000, 3).astype(np.float32)
>>> b = np.random.random(30000).reshape(10000, 3).astype(np.float32)
# All reside within a 1 x 1 x 1 box
>>> box = np.ones(3).astype(np.float32)

# Find all pairs witin 0.25 of each other
# Returns indices of pairs and the distance between them
>>> capped_distance_array(a, b, 0.25, box)
(array([[ 226, 8896],
        [ 226, 3557],
        [ 226, 8982],
        ..., 
        [ 259,   11],
        [ 259, 2215],
        [ 259, 5117]]),
 array([ 0.21252801,  0.21431111,  0.12156317, ...,  0.02756999,
         0.24850761,  0.15750615], dtype=float32))

```

Internally this is done using the eponymous CellGrid object, which takes coordinates and sorts them spatially into Cells.

``` python
>>> from cellgrid import CellGrid

>>> cg = CellGrid(box, 0.25, a)
>>> cg
<CellGrid with dimensions 4, 4, 4>
>>> cg[0]
<Cell at (0, 0, 0) with 13 coords>

```

