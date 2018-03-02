import numpy as np
from gaupro import kernels


def predict(x_star, sigma_n, A, X, y):
    a_inv = np.linalg.inv(A)
    mean_pred = (sigma_n ** (-2)) * np.dot(np.dot(np.dot(x_star.T, a_inv), X), y.T)
    sigma_pred = np.dot(np.dot(x_star.T, a_inv), x_star)
    return mean_pred, sigma_pred


def kernel_matrix(kernel, x1, x2=None):
    if x2 is None:
        x2 = x1

    n1 = x1.shape[1]
    n2 = x2.shape[1]
    matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            matrix[i, j] = kernel(x1[:, i], x2[:, j])
    return matrix


def black_box1(x):
    return x * np.sin(x)


def black_box2(x):
    a = 2.8
    return 0.1 * (-(x - a) ** 4 + (x - a) ** 3 + 10 * (x - a) ** 2)


def black_box3(x, axis):
    sigma1 = 2
    sigma2 = 1
    mu1 = 1
    mu2 = -1
    # return np.exp(-np.linalg.norm(x, axis=axis) / sigma)
    mode1 = np.exp(-np.linalg.norm(x - mu1, axis=axis) / sigma1)
    mode2 = np.exp(-np.linalg.norm(x - mu2, axis=axis) / sigma2)
    return mode1 + mode2


def black_box4(x):
    a = 10
    return 1 / (1 + np.exp(-a*(x - 3)))


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


def erf(x):
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x**2)
    return sign * y  # erf(-x) = -erf(x)


def normal_cdf(x):
    return (1 / 2) * (1 + erf(x / np.square(2)))
