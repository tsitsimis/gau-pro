import numpy as np
from carly import kernels


def predict(x_star, sigma_n, A, X, y):
    a_inv = np.linalg.inv(A)
    mean_pred = (sigma_n ** (-2)) * np.dot(np.dot(np.dot(x_star.T, a_inv), X), y.T)
    sigma_pred = np.dot(np.dot(x_star.T, a_inv), x_star)
    return mean_pred, sigma_pred


def kernel_matrix(kernel, x1, x2=None):
    if x2 is None:
        x2 = x1

    if kernel == 'se':
        ker = kernels.se_kernel(1.0)
    elif kernel == 'min':
        ker = kernels.min_kernel()

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            matrix[i, j] = ker(x1[i], x2[j])
    return matrix


def black_box1(x):
    return x * np.sin(x)


def black_box2(x):
    a = 2.8
    return 0.1 * (-(x - a) ** 4 + (x - a) ** 3 + 10 * (x - a) ** 2)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def weights_post_pdf(grid, X1, X2, y1, y2, sigma_prior):
    f1 = np.dot(grid, X1)
    f2 = np.dot(grid, X2)
    pdf = np.sum(logistic(y1 * f1), axis=2) + np.sum(logistic(y2 * f2), axis=2)

    sigma_prior_sqrt = sigma_prior ** 0.5
    penalty = np.dot(grid, sigma_prior_sqrt)
    penalty = np.linalg.norm(penalty, axis=2)
    pdf = pdf - 0.5 * penalty
    pdf = np.exp(pdf)
    pdf = pdf / np.sum(pdf)
    return pdf


def lin_reg_predict(x_star, w_grid, X1, X2, y1, y2, sigma_prior):
    p = logistic(np.dot(w_grid, x_star)) * weights_post_pdf(w_grid, X1, X2, y1, y2, sigma_prior)
    p = np.sum(p)
    return p
