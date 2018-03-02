import numpy as np
from gaupro import utils as uu


TINY = 1e-5


class Regressor:
    def __init__(self, kernel, sigma_n=0):
        # self.X_test = X_test
        # self.n_test = X_test.shape[1]
        # self.dim = X_test.shape[0]

        self.kernel = kernel

        self.sigma_n = sigma_n
        self.X_train = None
        self.y_train = None
        self.X_test = None

        self.mu = None
        self.cov = None
        self.cov_train_train = None
        self.cov_train_train_inv = None
        self.cov_test_train = None
        self.cov_train_test = None
        self.cov_test_test = None

    def fit(self, X, y):
        if X is not None:
            self.X_train = X

        if y is not None:
            self.y_train = y

        if (self.X_train is None) or (self.y_train is None):
            print('Error: No train set is specified. Call the function fit() with arguments.')
            return

        self.cov_train_train = uu.kernel_matrix(self.kernel, self.X_train) + self.sigma_n ** 2 * np.eye(self.X_train.shape[1])
        self.cov_train_train += TINY * np.eye(self.cov_train_train.shape[0])  # avoid singularities
        self.cov_train_train_inv = np.linalg.inv(self.cov_train_train)

    def predict(self, X):
        self.X_test = X
        self.cov_test_test = uu.kernel_matrix(self.kernel, self.X_test)
        self.cov_test_train = uu.kernel_matrix(self.kernel, self.X_test, self.X_train)
        self.cov_train_test = self.cov_test_train.T

        self.mu = np.dot(np.dot(self.cov_test_train, self.cov_train_train_inv), self.y_train)
        self.cov = self.cov_test_test - self.cov_test_train.dot(self.cov_train_train_inv.dot(self.cov_train_test))

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
        self.X_train = np.concatenate((self.X_train, x_new_vec), axis=1)
        self.y_train = np.concatenate((self.y_train, y_new_vec), axis=0)

