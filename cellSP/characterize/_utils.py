import pandas as pd
import numpy as np
import multiprocessing as mp
import operator as op
import functools
import collections
import tempfile
import random
import math
import sys
import os
import logging

#Taken from https://stackoverflow.com/questions/22229796/choose-at-random-from-combinations
#Adapted from https://github.com/salzman-lab/SPRAWL/blob/main/package/src/sprawl/utils.py.
def random_combination(iterable, r):
    """
    Random selection from itertools.combinations(iterable, r)
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_mean_pairs_dist(spots, num_pairs):
    """
    Helper function to choose 'num_pairs' pairs of gene spots and calculate the mean distance for each gene
    Input is an array of spots that it will choose from
    """
    d = 0
    for _ in range(num_pairs):
        (x1,y1),(x2,y2) = random_combination(spots,2)
        d += math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

    return d/num_pairs

def random_mean_pairs_angle(spots, centroid, num_pairs):
    """
    Helper function to choose 'num_pairs' pairs of gene spots and calculate the mean angle for each gene
    Input is an array of spots that it will choose from and the cell centroid as a tuple of (x,y)
    """
    cx,cy = centroid
    ang_sum = 0
    for _ in range(num_pairs):
        (x1,y1),(x2,y2) = random_combination(spots,2)
        v1 = (x1-cx,y1-cy)
        v2 = (x2-cx,y2-cy)
        ang_sum += angle_between(v1, v2)

    return ang_sum/num_pairs


#taken directly from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

