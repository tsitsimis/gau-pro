import numpy as np


def se_kernel(scale):
    def __se_kernel(x, y):
        return np.exp(-0.5 * np.linalg.norm(x - y) ** 2 / scale)
    return __se_kernel
