import numpy as np


def mu_plus_cov(kappa):
    return lambda mu, cov: mu + kappa * cov[np.diag_indices_from(cov)]


def mu_plus_cov_proba(mu, cov):
    kappa = 0.2
    beta = 2
    pdf = mu + kappa * cov[np.diag_indices_from(cov)]
    pdf = np.exp(beta * pdf)
    pdf = pdf / np.sum(pdf)
    return pdf
