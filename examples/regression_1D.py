import gaupro as gp
import gaupro.utils as uu
import gaupro.kernels as kers
import numpy as np
import matplotlib.pyplot as plt


# train inputs
d = 1
x_lim = [0, 20]
n_train = 10
x_train = np.random.uniform(x_lim[0], x_lim[1], n_train).reshape((d, n_train))
y_train = uu.black_box1(x_train).reshape((n_train, 1))
sigma_n = 0.0
noise = np.random.normal(0, sigma_n, n_train).reshape((n_train, 1))
y_train += noise

# test inputs
n_test = 200
x_test = np.linspace(x_lim[0], x_lim[1], n_test).reshape((d, n_test))

# fit
regressor = gp.Regressor(kers.se_kernel())
regressor.fit(x_train, y_train)
regressor.predict(x_test)

# plot
t = np.linspace(x_lim[0], x_lim[1], 100)
plt.plot(t, uu.black_box1(t), c='k', linestyle=':')

plt.scatter(x_train, y_train, marker='+', c='r', s=120, zorder=10)
plt.plot(x_test[0], regressor.mu.T[0], c='k', zorder=10)
plt.fill_between(x_test[0], regressor.mu.T[0] - 3 * np.sqrt(regressor.cov[np.diag_indices_from(regressor.cov)]),
                 regressor.mu.T[0] + 3 * np.sqrt(regressor.cov[np.diag_indices_from(regressor.cov)]),
                 facecolor='gray', alpha=0.5)
plt.show()
