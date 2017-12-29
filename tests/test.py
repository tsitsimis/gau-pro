import carly
import carly.utils as uu
import numpy as np
import matplotlib.pyplot as plt


# train inputs
x_lim = [0, 20]
n_train = 10
x_train = np.random.uniform(x_lim[0], x_lim[1], n_train)
f_train = uu.objective(x_train)
sigma_n = 0.0
noise = np.random.normal(0, sigma_n, n_train)
f_train += noise

# test inputs
n_test = 200
x_test = np.linspace(x_lim[0], x_lim[1], n_test)

# fit
model = carly.Regression('se', sigma_n=sigma_n)
model.fit(x_train, f_train, x_test)

# pick samples
samples = model.pick_samples(1)

# plot
t = np.linspace(x_lim[0], x_lim[1], 100)
plt.plot(t, uu.objective(t), c='b')
plt.scatter(x_train, f_train, marker='+', c='r', s=120, zorder=10)
plt.scatter(x_test, model.mu, marker='.', facecolors='none', edgecolors='k', linewidth=0.8, zorder=10)
plt.show()
