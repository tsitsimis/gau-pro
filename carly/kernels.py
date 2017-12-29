import numpy as np


def se_kernel(x, y):
    return np.exp(-0.5 * np.linalg.norm(x - y) ** 2 / 1.0)
