import math

import numpy as np

from act_functions import leaky_relu, relu, sigmoid


class Layer:
    
    def __init__(self, input_size, n_neurons, function) -> None:
        self.matrix = np.random.normal(0, 1 / math.sqrt(input_size), (input_size, n_neurons))
        self.biases = np.zeros((1, n_neurons))
        self.function = self.__get_function(function)
        
    def __get_function(self, function):
        if function == 'sigmoid':
            return sigmoid
        elif function == 'relu':
            return relu
        elif function == 'leaky_relu':
            return leaky_relu
        else:
            raise ValueError(f"Unsupported activation function: {function}")
    
        
    
    def set_output_z(self, output):
        self.output_z = output
        
    def set_input(self, input_x):
        self.input_x = input_x
        
    def set_output_a(self, output):
        self.output_a = output
        
    