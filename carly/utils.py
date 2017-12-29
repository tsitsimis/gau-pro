import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def predict(x_star, sigma_n, A, X, y):
    a_inv = np.linalg.inv(A)
    mean_pred = (sigma_n ** (-2)) * np.dot(np.dot(np.dot(x_star.T, a_inv), X), y.T)
    sigma_pred = np.dot(np.dot(x_star.T, a_inv), x_star)
    return mean_pred, sigma_pred


def se_matrix(x1, x2=None, scale=1):
    n1 = np.shape(x1)[0]
    if x2 is None:
        matrix = np.zeros((n1, n1))
        for i in range(n1):
            for j in range(n1):
                matrix[i, j] = np.exp(-0.5 * np.linalg.norm(x1[i] - x1[j]) ** 2 / scale)
        return matrix

    n2 = np.shape(x2)[0]
    matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            matrix[i, j] = np.exp(-0.5 * np.linalg.norm(x1[i] - x2[j]) ** 2 / scale)
    return matrix


def min_matrix(x1):
    n1 = np.shape(x1)[0]
    matrix = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n1):
            matrix[i, j] = np.min([x1[i], x1[j]]) - x1[i] * x1[j]
    return matrix


def objective(x):
    return x * np.sin(x)


def plot_mvn():
    w1 = np.linspace(-5, 5, 100)
    w2 = np.linspace(-5, 5, 100)
    W1, W2 = np.meshgrid(w1, w2)
    grid = np.empty(W1.shape + (2,))
    grid[:, :, 0] = W1
    grid[:, :, 1] = W2

    x1 = 3
    x2 = 3.1
    cov = [[1, np.exp(-0.5 * (x1 - x2) ** 2)], [np.exp(-0.5 * (x1 - x2) ** 2), 1]]
    pdf = multivariate_normal([0, 0], cov)

    plt.contour(W1, W2, pdf.pdf(grid), 3, colors='k')
    plt.grid(alpha=0.5)
    plt.show()


def plot_gp_samples(x_test, scale=1, n_samples=1):
    n_test = x_test.shape[0]

    mu = np.zeros(n_test)
    cov = se_matrix(x_test, scale=scale)
    samples = np.random.multivariate_normal(mu, cov, n_samples)

    for i in range(n_samples):
        plt.plot(x_test, samples[i, :], marker='')


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
