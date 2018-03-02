import numpy as np


def se_kernel(scale=1.0):
    def __kernel(x, y):
        return np.exp(-0.5 * np.linalg.norm(x - y) ** 2 / scale)
    return __kernel


def min_kernel():
    def __kernel(x, y):
        return np.min([x, y])
    return __kernel


def linear_kernel():
    def __kernel(x, y):
        return np.dot(x.T, y)
    return __kernel


def poly_kernel(p):
    def __kernel(x, y):
        return (np.dot(x.T, y) + 1) ** p
    return __kernel


def constant_kernel(c):
    def __kernel(x, y):
        return c
    return __kernel


def ou_kernel(scale):
    def __kernel(x, y):
        return np.exp(-np.linalg.norm(x - y) / scale)
    return __kernel


def periodic_kernel(scale):
    def __kernel(x, y):
        return np.exp(-(2 / scale**2) * (np.sin((x - y) / 2))**2 / scale)
    return __kernel


def rq_kernel(a):
    def __kernel(x, y):
        return (1 + (np.linalg.norm(x - y))**2)**(-a)
    return __kernel
