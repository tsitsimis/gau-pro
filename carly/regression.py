import numpy as np
from carly import utils as uu


class Regression:
    def __init__(self, kernel, sigma_n=0):
        self.kernel = kernel

        self.sigma_n = sigma_n
        self.mu = None
        self.cov = None
        self.X_train = None
        self.y_train = None
        self.X_test = None

        self.acquisition_function = None
        self.acq = None

    def fit(self, X_train, y_train, X_test):
        if X_train is not None:
            self.X_train = X_train

        if y_train is not None:
            self.y_train = y_train

        if X_test is not None:
            self.X_test = X_test

        if (self.X_train is None) or (self.y_train is None) or (self.X_test is None):
            print('Error: No train or test set is specified. Call the function fit() with arguments.')
            return

        cov_test_test = uu.kernel_matrix(self.kernel, self.X_test)
        cov_train_train = uu.kernel_matrix(self.kernel, self.X_train) + self.sigma_n**2 * np.eye(self.X_train.shape[0])
        cov_test_train = uu.kernel_matrix(self.kernel, self.X_test, self.X_train)
        cov_train_test = cov_test_train.T
        cov_train_train_inv = np.linalg.inv(cov_train_train)

        self.mu = np.dot(np.dot(cov_test_train, cov_train_train_inv), y_train)
        self.cov = cov_test_test - np.dot(np.dot(cov_test_train, cov_train_train_inv), cov_train_test)

        if self.acquisition_function is not None:
            self.acq = self.acquisition_function(self.mu, self.cov)

    def pick_samples(self, n_samples):
        if (self.mu is None) or (self.cov is None):
            print('Error: The model is not fitted. Call the function fit() before sampling the GP.')
            return

        samples = np.random.multivariate_normal(self.mu, self.cov, n_samples)
        samples = np.reshape(samples, (self.mu.shape[0], n_samples))
        return samples

    def augment_train(self, x_new, y_new):
        self.X_train = np.append(self.X_train, [x_new])
        self.y_train = np.append(self.y_train, [y_new])

    def set_acquisition(self, acq_func):
        self.acquisition_function = acq_func
        self.acq = self.acquisition_function(self.mu, self.cov)
