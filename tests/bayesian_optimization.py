import numpy as np
import carly
import carly.utils as uu
import carly.kernels as kernels
import carly.baysian_optimizer as boplt
import carly.acquisition_functions as acq


# train inputs
x_lim = [0, 6.5]
n_train = 1
np.random.seed(42)
x_train = np.random.uniform(x_lim[0], x_lim[1], n_train)
f_train = uu.objective2(x_train)
sigma_n = 0.0
noise = np.random.normal(0, sigma_n, n_train)
f_train += noise

# test inputs
n_test = 200
x_test = np.linspace(x_lim[0], x_lim[1], n_test)

# fit
model = carly.Regression(kernels.se_kernel(scale=1), sigma_n=sigma_n)
model.fit(x_train, f_train, x_test)

# pick samples
# samples = model.pick_samples(1)

# acquisition function
model.set_acquisition(acq.simple(1.0))
# kappa = 1
# acq_fun = model.mu + kappa * model.cov[np.diag_indices_from(model.cov)]
# model.acq = acq_fun
# acq_fun = acq_fun - np.min(acq_fun)
# acq_fun = acq_fun / np.sum(acq_fun)

ploter = boplt.BayesianOptPlot(model)
ploter.draw()
