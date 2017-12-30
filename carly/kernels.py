import numpy as np


def se_kernel(scale):
    return lambda x, y: np.exp(-0.5 * np.linalg.norm(x - y) ** 2 / scale)
