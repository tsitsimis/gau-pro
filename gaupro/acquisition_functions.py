import numpy as np
import gaupro.utils as uu


# mu + k*sigma
def mu_plus_cov(kappa):
    def __acq_func(mu, cov):
        return mu + kappa * cov[np.diag_indices_from(cov)]
    return __acq_func


# Probability of Improvement
def PI(ksi):
    def __acq_func(mu, cov):
        mu_plus = np.max(mu).reshape((1, 1))
        n_test = cov[np.diag_indices_from(cov)].shape[0]
        cov_diag = cov[np.diag_indices_from(cov)].reshape((n_test, 1))
        return uu.normal_cdf((mu - mu_plus - ksi) / cov_diag)
    return __acq_func
