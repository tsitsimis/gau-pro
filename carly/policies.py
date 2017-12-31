import numpy as np


def max_policy(acq):
    return np.argmax(acq)


def proba_policy(beta):
    def __proba_policy(acq):
        pdf = acq
        pdf = np.exp(beta * pdf)
        pdf = pdf / np.sum(pdf)
        return np.random.choice(acq.shape[0], 1, p=pdf)[0]
    return __proba_policy
