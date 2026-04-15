import numpy as np



def step_function(z):
    return 1 if z > 0 else 0


def sigmoid(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        return a * (1 - a)
    return a


def relu(z, derivative=False):
    if derivative:
        return np.where(z > 0, 1, 0)
    return np.where(z > 0, z, 0)


def leaky_relu(z, derivative=False):
    if derivative:
        return np.where(z > 0, 1, 0.01)
    return np.where(z > 0, z, 0.01 * z)
