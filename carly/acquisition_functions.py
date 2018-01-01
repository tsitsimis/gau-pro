import numpy as np
import carly.utils as uu


# μ + κ*σ
def mu_plus_cov(kappa):
    def __acq_func(mu, cov):
        return mu + kappa * cov[np.diag_indices_from(cov)]
    return __acq_func


# Probability of Improvement
def PI(xi):
    def __acq_func(mu, cov):
        mu_plus = np.max(mu)
        return uu.normal_cdf((mu - mu_plus - xi) / cov[np.diag_indices_from(cov)])
    return __acq_func
