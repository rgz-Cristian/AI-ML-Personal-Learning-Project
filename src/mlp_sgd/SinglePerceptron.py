import math
import numpy as np

from . import funcs



class SinglePerceptron:

    def __init__(self, n_input, learning_rate, activation=funcs.sigmoid) -> None:
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 1 / math.sqrt(n_input), n_input)
        self.bias = 0
        self.activation = activation

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.__f_activation(z)

    def __f_activation(self, z):
        return self.activation(z)

    def fit(self, x, y):

        y_pred = self.predict(x)

        err = y - y_pred

        delta_w = self.learning_rate * err * x
        self.weights = self.weights + delta_w

        delta_b = self.learning_rate * err
        self.bias = self.bias + delta_b

    def fit_error(self, x, err):

        delta_w = self.learning_rate * err * x
        self.weights = self.weights + delta_w

        delta_b = self.learning_rate * err
        self.bias = self.bias + delta_b
