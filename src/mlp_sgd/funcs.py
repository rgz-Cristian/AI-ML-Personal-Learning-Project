import numpy as np


def step_function(z):
    return 1 if z > 0 else 0


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def sigmoid_der(z):
    return z * (1 - z)


def relu(z):
    return np.where(z > 0, z, 0)


def relu_der(z):
    return np.where(z > 0, 1, 0)


def leaky_relu(z):
    return np.where(z > 0, z, 0.01 * z)


def leaky_relu_der(z):
    return np.where(z > 0, 1, 0.01)
