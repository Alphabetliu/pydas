# -*- coding: utf-8 -*-

import warnings
import numpy as np


def diff1d(y, dx):
    """Calculate the first-order derivative of the signal y.
    @param: y - the signal
    @param: dx - interval"""

    y = np.asarray(y, dtype='float')
    dy = np.zeros_like(y)

    if y.shape[0] <= 5:
        warnings.warn("Array size is too small!")
        return dy
    else:
        dy[0] = (-y[2] + 4 * y[1] - 3 * y[0]) / (2 * dx)
        dy[1] = (-y[3] + 6 * y[2] - 3 * y[1] - 2 * y[0]) / (6 * dx)
        dy[2] = (8 * (y[3] - y[1]) - (y[4] - y[0])) / (12 * dx)
        dy[3:-3] = (45 * (y[4:-2] - y[2:-4]) - 9 *
                    (y[5:-1] - y[1:-5]) + (y[6:] - y[:-6])) / (60 * dx)
        dy[-3] = (8 * (y[-2] - y[-4]) - (y[-1] - y[-5])) / (12 * dx)
        dy[-2] = (2 * y[-1] + 3 * y[-2] - 6 * y[-3] + y[-4]) / (6 * dx)
        dy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dx)

        return dy
