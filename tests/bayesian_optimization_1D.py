import numpy as np
import carly
import carly.utils as uu
import carly.kernels as kernels
import carly.acquisition_functions as acq
import carly.policies as pol
import matplotlib.pyplot as plt

# train inputs
input_lim = [0, 6.5]
n_train = 10
d = 1
# np.random.seed(42)
X = np.random.uniform(input_lim[0]+3.1, input_lim[1], n_train)
X = np.reshape(X, (d, n_train))
# print(X.shape)

black_box = uu.black_box4
y = black_box(X).reshape((n_train, 1))

# test inputs
n_test = 200
x_test = np.linspace(input_lim[0], input_lim[1], n_test).reshape((d, n_test))

# model
ker = kernels.se_kernel(0.3)
model = carly.Regression(x_test, ker)
model.fit(X, y)

# optimize + animate
optimizer = carly.BayesianOptimizer(model, black_box,
                                    acquisition_func=acq.PI(0.001),
                                    policy=pol.max_policy)

fig, ax = plt.subplots(2, 1)
plt.ion()
t = np.linspace(input_lim[0], input_lim[1], 100)

for i in range(1):
    # plot
    ax[0].cla()
    ax[1].cla()

    ax[0].plot(t, black_box(t), c='k', linestyle=':')
    ax[0].scatter(optimizer.history[0, :], optimizer.history[1, :], marker='+', c='r', s=120, zorder=10)
    ax[0].plot(model.X_test.T, model.mu, c='k', zorder=10)
    ax[0].fill_between(model.X_test[0, :], model.mu[:, 0] - 2 * np.sqrt(np.abs(model.cov[np.diag_indices_from(model.cov)])),
                       model.mu[:, 0] + 2 * np.sqrt(np.abs(model.cov[np.diag_indices_from(model.cov)])),
                       facecolor='gray',
                       alpha=0.5)
    # ax[0].fill_between(model.X_test[0, :], model.mu-1,
    #                    model.mu+1,
    #                    facecolor='gray',
    #                    alpha=0.5)

    ax[1].plot(model.X_test.T, optimizer.acquisition_func(model.mu, model.cov), c='g', linewidth=2)
    if optimizer.i_next is not None:
        ax[1].scatter(optimizer.x_next, optimizer.acquisition_func(model.mu, model.cov)[optimizer.i_next],
                      marker='v', c='r', s=120, zorder=10)

    optimizer.update()

    plt.show()
    plt.pause(1)

plt.ioff()
plt.show()
