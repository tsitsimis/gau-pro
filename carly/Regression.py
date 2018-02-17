import numpy as np
from carly import utils as uu


TINY = 1e-5


class Regression:
    def __init__(self, X_test, kernel, sigma_n=0):
        self.X_test = X_test
        self.n_test = X_test.shape[1]
        self.dim = X_test.shape[0]

        self.kernel = kernel

        self.sigma_n = sigma_n
        self.X = np.array([])
        self.y = np.array([])

        # fit without any train inputs
        self.mu = np.zeros((self.n_test, 1))
        self.cov = np.eye(self.n_test)

    def fit(self, X, y):
        if X is not None:
            self.X = X

        if y is not None:
            self.y = y

        if (self.X is None) or (self.y is None):
            print('Error: No train set is specified. Call the function fit() with arguments.')
            return

        cov_test_test = uu.kernel_matrix(self.kernel, self.X_test)
        cov_train_train = uu.kernel_matrix(self.kernel, self.X) + self.sigma_n**2 * np.eye(self.X.shape[1])
        cov_train_train += TINY * np.eye(cov_train_train.shape[0])  # avoid singularities
        cov_test_train = uu.kernel_matrix(self.kernel, self.X_test, self.X)
        cov_train_test = cov_test_train.T
        cov_train_train_inv = np.linalg.inv(cov_train_train)

        self.mu = np.dot(np.dot(cov_test_train, cov_train_train_inv), self.y)
        self.cov = cov_test_test - np.dot(np.dot(cov_test_train, cov_train_train_inv), cov_train_test)

    def pick_samples(self, n_samples):
        if (self.mu is None) or (self.cov is None):
            print('Error: The model is not fitted. Call the function fit() before sampling the GP.')
            return

        samples = np.random.multivariate_normal(self.mu.T[0], self.cov, n_samples)
        samples = np.reshape(samples, (self.mu.shape[0], n_samples))
        return samples

    def augment_train(self, x_new, y_new):
        x_new_vec = np.array([[x_new]])
        y_new_vec = np.array([[y_new]])
        self.X = np.concatenate((self.X, x_new_vec), axis=1)
        self.y = np.concatenate((self.y, y_new_vec), axis=0)

