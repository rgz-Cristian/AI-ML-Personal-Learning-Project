import numpy as np
from SinglePerceptron import SinglePerceptron
import funcs


class Layer:

    def __init__(
        self, n_input, n_neurons, learning_rate, function="leaky_relu"
    ) -> None:
        self.n_inputs = n_input
        self.learning_rate = learning_rate
        self.function = function
        self.neurons = self.__init_neurons_list(n_input, n_neurons, learning_rate)

    def __init_neurons_list(self, n_input, n_neurons, learning_rate):
        array_neurons = list()
        for i in range(n_neurons):
            neuron = SinglePerceptron(
                n_input,
                learning_rate,
                activation=self._select_activation_function(self.function),
            )
            array_neurons.append(neuron)

        return array_neurons

    def _select_activation_function(self, function):
        if function == "leaky_relu":
            return funcs.leaky_relu
        elif function == "relu":
            return funcs.relu
        elif function == "sigmoid":
            return funcs.sigmoid
        else:
            raise ValueError("Invalid activation function")

    def predict(self, x):
        output_array = np.array([])

        for n in self.neurons:
            y_pred = n.predict(x)
            output_array = np.append(output_array, y_pred)

        return output_array

    def fit(self, x, errors):
        for n, err in zip(self.neurons, errors):
            n.fit_error(x, err)

    def get_weights(self):
        weights_matrix = np.array([n.weights for n in self.neurons])

        return weights_matrix
