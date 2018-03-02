import numpy as np


class BayesianOptimizer:
    def __init__(self, model, black_box, acquisition_func, policy):
        self.model = model
        self.black_box = black_box
        self.acquisition_func = acquisition_func
        self.policy = policy

        self.__find_next_x()

        self.history = np.empty((2, 0))
        # if the model is fitted already, add the train inputs to optimizer's history
        model_history = np.concatenate((self.model.X, self.model.y.T))
        self.history = np.concatenate((self.history, model_history), axis=1)

        y_next = self.black_box(self.x_next)
        self.history = np.append(self.history, np.array([[self.x_next], [y_next]]), axis=1)

    def update(self):
        y_next = self.black_box(self.x_next)           # sample the black box function
        self.model.augment_train(self.x_next, y_next)  # augment the data
        self.model.fit(self.model.X, self.model.y)     # update the GP

        self.history = np.append(self.history, np.array([[self.x_next], [y_next]]), axis=1)
        self.__find_next_x()  # find next x

    def __find_next_x(self):
        i_next = self.policy(self.acquisition_func(self.model.mu, self.model.cov))
        x_next = self.model.X_test[0, i_next]
        self.i_next = i_next
        self.x_next = x_next
