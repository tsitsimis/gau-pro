import numpy as np


def mu_plus_cov(kappa):
    def __mu_plus_cov(mu, cov):
        return mu + kappa * cov[np.diag_indices_from(cov)]
    return __mu_plus_cov
