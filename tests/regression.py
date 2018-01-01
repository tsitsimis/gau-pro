import carly
import carly.utils as uu
import numpy as np
import matplotlib.pyplot as plt


# train inputs
x_lim = [0, 20]
n_train = 10
x_train = np.random.uniform(x_lim[0], x_lim[1], n_train)
f_train = uu.black_box1(x_train)
sigma_n = 0.0
noise = np.random.normal(0, sigma_n, n_train)
f_train += noise

# test inputs
n_test = 200
x_test = np.linspace(x_lim[0], x_lim[1], n_test)

# fit
model = carly.Regression(x_test, 'min', sigma_n=sigma_n)
model.fit(x_train, f_train)

# pick samples
samples = model.pick_samples(1)

# plot
t = np.linspace(x_lim[0], x_lim[1], 100)
plt.plot(t, uu.black_box1(t), c='k', linestyle=':')

plt.scatter(x_train, f_train, marker='+', c='r', s=120, zorder=10)
plt.plot(x_test, model.mu, c='k', zorder=10)
plt.show()
