import gaupro
import gaupro.utils as uu
import gaupro.kernels as kers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


# train inputs
d = 2
x_lim = [-3, 3]
n_train = 200
x_train = (x_lim[1] - x_lim[0]) * np.random.random((n_train, d)).T + x_lim[0]
f_train = uu.black_box3(x_train, axis=0)

# test inputs
n_test_x = 20
n_test_y = 20

x_test_x = np.linspace(x_lim[0], x_lim[1], n_test_x).reshape((1, n_test_x))
x_test_y = np.linspace(x_lim[0], x_lim[1], n_test_y).reshape((1, n_test_y))

X, Y = np.meshgrid(x_test_x, x_test_y)
X_vec = np.resize(X, (1, n_test_x * n_test_y))
Y_vec = np.resize(Y, (1, n_test_x * n_test_y))

x_test = np.concatenate((X_vec, Y_vec), axis=0)

# fit
model = gaupro.Regressor(x_test, kers.se_kernel(1.0))
model.fit(x_train, f_train)
mu_vec = model.mu.reshape((n_test_x, n_test_y))

# pick samples
# samples = model.pick_samples(1)

# fig = plt.figure()
# ax = Axes3D(fig)
fig, ax = plt.subplots(1, 2)

# objective
no = 40
Xo, Yo = np.meshgrid(np.linspace(x_lim[0], x_lim[1], no), np.linspace(x_lim[0], x_lim[1], no))
grid = np.empty((Xo.shape[0], Xo.shape[0], 2))
grid[:, :, 0] = Xo
grid[:, :, 1] = Yo
Zo = uu.black_box3(grid, axis=2)
# ax.plot_wireframe(Xo, Yo, Zo, alpha=0.5, zorder=1, rstride=1, cstride=1, linestyle='-', color='r')
ax[0].contour(Xo, Yo, Zo, 20)

# train
# ax.scatter(x_train[0, :], x_train[1, :], zs=f_train, marker='.', c='r', s=80, zorder=10)
ax[0].scatter(x_train[0, :], x_train[1, :], marker='x', c='r', s=20, zorder=10)

# test
# ax.plot_wireframe(X, Y, mu_vec, rstride=1, cstride=1, zorder=1)
ax[1].contour(X, Y, mu_vec, 20)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

ax[0].axis('square')
ax[0].grid(linestyle=':', alpha=0.8, markevery=20)
ax[1].axis('square')
ax[1].grid(linestyle=':', alpha=0.8, markevery=20)
plt.show()
