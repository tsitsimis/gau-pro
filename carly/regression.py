import numpy as np
from carly import utils as uu


class Regression:
    def __init__(self, kernel, sigma_n=0):
        if kernel == 'sqexp':
            self.kernel = uu.se_matrix

        self.sigma_n = sigma_n
        self.mu = None
        self.cov = None

    def fit(self, X_train, y_train, X_test):
        scale = 1
        cov_test_test = self.kernel(X_test, scale=scale)
        cov_train_train = self.kernel(X_train, scale=scale) + self.sigma_n**2 * np.eye(X_train.shape[0])
        cov_test_train = self.kernel(X_test, X_train, scale=scale)
        cov_train_test = cov_test_train.T
        cov_train_train_inv = np.linalg.inv(cov_train_train)

        self.mu = np.dot(np.dot(cov_test_train, cov_train_train_inv), y_train)
        self.cov = cov_test_test - np.dot(np.dot(cov_test_train, cov_train_train_inv), cov_train_test)

    def pick_samples(self, n_samples):
        if (self.mu is None) or (self.cov is None):
            print('Error: The model is not fitted. Call the function fit() before sampling the GP.')
            return

        samples = np.random.multivariate_normal(self.mu, self.cov, n_samples)
        samples = np.reshape(samples, (self.mu.shape[0], n_samples))
        return samples
