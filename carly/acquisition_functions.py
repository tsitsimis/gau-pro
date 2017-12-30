import numpy as np


def simple(kappa):
    return lambda mu, cov: mu + kappa * cov[np.diag_indices_from(cov)]
