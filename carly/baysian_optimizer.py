import numpy as np
import carly.utils as uu
import matplotlib.pyplot as plt


class BayesianOptPlot:
    def __init__(self, model):
        self.model = model

        # initialize figure
        fig, ax = plt.subplots(2, 1)
        self.fig = fig
        self.ax = ax

    def update(self):
        model = self.model

        # find next x
        i_next = np.argmax(model.acq)
        x_next = model.X_test[i_next] + np.random.normal(0, 0.01)  # avoid choosing the same x twice

        f_next = uu.objective2(x_next)                         # sample the objective function
        model.augment_train(x_next, f_next)                    # augment the data
        model.fit(model.X_train, model.y_train, model.X_test)  # update the GP

        self.model = model
        self.draw()

    def draw(self):
        model = self.model
        fig = self.fig
        ax = self.ax

        ax[0].cla()
        ax[1].cla()

        fig.canvas.mpl_connect('key_press_event', self.press)

        t = np.linspace(np.min(model.X_test), np.max(model.X_test), 100)
        ax[0].plot(t, uu.objective2(t), c='b')

        # train data
        ax[0].scatter(model.X_train, model.y_train, marker='+', c='r', s=120, zorder=10)

        # mean + var
        ax[0].plot(model.X_test, model.mu, c='k', zorder=10)
        ax[0].fill_between(model.X_test, model.mu - 2 * np.sqrt(model.cov[np.diag_indices_from(model.cov)]),
                           model.mu + 2 * np.sqrt(model.cov[np.diag_indices_from(model.cov)]),
                           facecolor='gray',
                           alpha=0.5)

        # acquisition functions
        ax[1].plot(model.X_test, model.acq, c='green', linewidth=1.5, zorder=20)
        plt.show()

    def press(self, event):
        if event.key == 'enter':
            self.update()
