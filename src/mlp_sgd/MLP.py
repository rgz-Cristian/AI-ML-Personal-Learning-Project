import numpy as np

from .Layer import Layer
from . import funcs



class SgdMLP:
    
    
    def __init__(self, layers, learning_rate) -> None:
        self.layers_list = self.__init_layers(layers, learning_rate)
        self.learning_rate = learning_rate

    def __init_layers(self, layers, learning_rate):
        layers_list = list()
        for layer_par in layers:
            layer = Layer(
                layer_par[0], layer_par[1], learning_rate, function=layer_par[2]
            )
            layers_list.append(layer)

        return layers_list

    def predict(self, X):
        X_input = X
        self.y_output = list()
        self.y_output.append(X_input)

        for layer in self.layers_list:
            X_input = layer.predict(X_input)
            self.y_output.append(X_input)

        return X_input

    def backward(self, X, y):
        delta_list = self.delta_backward(y)

        for delta_l, layer, inputs in zip(delta_list, self.layers_list, self.y_output):
            layer.fit(inputs, delta_l)

    def delta_backward(self, y):
        delta_list = list()
        delta_l = (y - self.y_output[-1]) * funcs.sigmoid_der(self.y_output[-1])

        delta_list.append(delta_l)

        index = len(self.layers_list) - 2

        while index >= 0:
            delta_l = np.dot(
                self.layers_list[index + 1].get_weights().T, delta_l
            ) * funcs.leaky_relu_der(self.y_output[index + 1])
            delta_list.append(delta_l)

            index -= 1

        delta_list.reverse()
        return delta_list

    def train(self, X, y, epochs, error_tolerance=0.01):
        epoch_errors = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for x, target in zip(X, y):
                self.predict(x)
                loss = np.mean((target - self.y_output[-1]) ** 2)
                epoch_loss += loss
                self.backward(x, target)
            
            # Promedio de MSE global de la época
            mse = epoch_loss / len(X)
            epoch_errors.append(mse)
            
            if mse < error_tolerance:
                print(f"Last epoch: {epoch} with MSE: {mse}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}, MSE: {mse}")
        
        self.error = np.array(epoch_errors)
        return self.error
        




