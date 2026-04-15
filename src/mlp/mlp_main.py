import numpy as np

from layer import Layer



class MLP:
    
    def __init__(self, layers: list, learning_rate: float) -> None:
        self.layers = list()
        self.__init_layers(layers)
        self.learning_rate = learning_rate
        
        
    def __init_layers(self, layers):
        for layer in layers:
            layer = Layer(layer[0], layer[1], layer[2])
            self.layers.append(layer)
            
        
    def forward(self, X):
        X_input = X
        
        for layer in self.layers:
            z = np.dot(X_input, layer.matrix) + layer.biases
            
            layer.set_input(X_input)
            layer.set_output_z(z)
            y = layer.function(z)
            layer.set_output_a(y)
            
            X_input = y
        
            
            
        return X_input

    def train(self, X, y, epochs, batch_size, error_tolerance=0.05):
        
        
        for epoch in range(epochs):
            mse = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                
                mse = 0.5 * (np.sum((y_batch - self.forward(X_batch)) ** 2))

                self.backward(X_batch, y_batch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, MSE: {mse}")
                
            if mse < error_tolerance:
                print(f"Last epoch: {epoch} with MSE: {mse}")
                break
            

    def backward(self, X, y):
        
        # backpropagation for output layer
        
        z_last_layer = self.layers[-1].output_z
                
        delta = (y - self.layers[-1].function(z_last_layer)) * self.layers[-1].function(z_last_layer, True)
        

        a_last_layer = self.layers[-1].input_x if len(self.layers) > 1 else X
        
        self.layers[-1].matrix += self.learning_rate * np.dot(a_last_layer.T, delta)
        self.layers[-1].biases += self.learning_rate * np.sum(delta, axis=0, keepdims=True)

        
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            
            
            delta = np.dot(delta, self.layers[i + 1].matrix.T) * self.layers[i - 1].function(layer.output_z, True)
                
            layer.matrix += self.learning_rate * np.dot(layer.input_x.T, delta)
            layer.biases += self.learning_rate * np.sum(delta, axis=0, keepdims=True)
        
        
    
    def predict(self, X):
        return self.forward(X)
    

    def get_all_weights(self):
        weights = []
        for layer in self.layers:
            weights.append((layer.matrix, layer.biases))
        return weights
    
    def get_weight_at(self, layer_index):
        if layer_index < 0 or layer_index >= len(self.layers):
            print(f"Error: Layer index {layer_index} is out of bounds.")
        return (self.layers[layer_index].matrix, self.layers[layer_index].biases)
    
    
    
    
inputs_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs_bits = np.array([[0], [1], [1], [0]])

mlp = MLP(layers=[(2, 4, 'leaky_relu'),(4, 3, 'leaky_relu'), (3, 1, 'sigmoid')], learning_rate=0.5)
mlp.train(inputs_bits, outputs_bits, epochs=1000, batch_size=4, error_tolerance=0.01)

for i in range(len(inputs_bits)):
    print(f"Input: {inputs_bits[i]}, Predicted Output: {mlp.predict(inputs_bits[i])}, Actual Output: {outputs_bits[i]}")