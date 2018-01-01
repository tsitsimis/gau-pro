import numpy as np


def se_kernel(scale):
    def __kernel(x, y):
        return np.exp(-0.5 * np.linalg.norm(x - y) ** 2 / scale)
    return __kernel


def min_kernel():
    def __kernel(x, y):
        return np.min([x, y])
    return __kernel
