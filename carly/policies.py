import numpy as np


def max_policy():
    return lambda acq: np.argmax(acq)


def proba_policy():
    return lambda acq: np.random.choice(acq.shape[0], 1, p=acq)[0]
