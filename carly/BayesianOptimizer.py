import numpy as np
import carly.acquisition_functions as af
import carly.policies as pol


class BayesianOptimizer:
    def __init__(self, model, black_box, acquisition_func, policy):
        self.model = model
        self.black_box = black_box
        self.acquisition_func = acquisition_func
        self.policy = policy

        self.i_next = None
        self.x_next = None
        self.history = np.empty((0, 2))
        # if the model is fitted already, add the train inputs to optimizer's history
        model_history = np.concatenate(([self.model.X], [self.model.y])).T
        self.history = np.concatenate((self.history, model_history), axis=0)

    def update(self):
        self.__find_next_x()                                      # find next x
        y_next = self.black_box(self.x_next)                    # sample the black box function
        self.model.augment_train(self.x_next, y_next)           # augment the data
        self.model.fit(self.model.X, self.model.y)  # update the GP

        self.history = np.append(self.history, np.array([[self.x_next, y_next]]), axis=0)

    def __find_next_x(self):
        i_next = self.policy(self.acquisition_func(self.model.mu, self.model.cov))
        x_next = self.model.X_test[i_next]  # + np.random.normal(0, 0.0)  # avoid choosing the same x twice
        self.i_next = i_next
        self.x_next = x_next
