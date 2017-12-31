import numpy as np
import carly
import carly.utils as uu
import carly.kernels as kernels
import carly.acquisition_functions as acq
import carly.policies as pol
import matplotlib.pyplot as plt

# train inputs
input_lim = [0, 6.5]
n_train = 0
np.random.seed(42)
X = np.random.uniform(input_lim[0], input_lim[1], n_train)

black_box = uu.black_box2
y = black_box(X)

# test inputs
n_test = 200
x_test = np.linspace(input_lim[0], input_lim[1], n_test)

# model
ker = kernels.se_kernel(1.0)
model = carly.Regression(x_test, ker)
model.fit(X, y)

# optimize + animate
optimizer = carly.BayesianOptimizer(model, black_box,
                                    acquisition_func=acq.mu_plus_cov(kappa=1.0),
                                    policy=pol.proba_policy(beta=5.0))

fig, ax = plt.subplots(2, 1)
plt.ion()
t = np.linspace(input_lim[0], input_lim[1], 100)

for i in range(10):
    optimizer.update()

    # plot
    ax[0].cla()
    ax[1].cla()

    ax[0].plot(t, uu.black_box2(t), c='k', linestyle=':')
    ax[0].scatter(optimizer.history[:, 0], optimizer.history[:, 1], marker='+', c='r', s=120, zorder=10)
    ax[0].plot(x_test, model.mu, c='k', zorder=10)
    ax[0].fill_between(model.X_test, model.mu - 2 * np.sqrt(np.abs(model.cov[np.diag_indices_from(model.cov)])),
                       model.mu + 2 * np.sqrt(np.abs(model.cov[np.diag_indices_from(model.cov)])),
                       facecolor='gray',
                       alpha=0.5)

    ax[1].plot(x_test, optimizer.acquisition_func(model.mu, model.cov), c='g', linewidth=2)
    if optimizer.i_next is not None:
        ax[1].scatter(optimizer.x_next, optimizer.acquisition_func(model.mu, model.cov)[optimizer.i_next],
                      marker='v', c='r', s=120, zorder=10)
    plt.show()
    plt.pause(0.2)

plt.ioff()
plt.show()
