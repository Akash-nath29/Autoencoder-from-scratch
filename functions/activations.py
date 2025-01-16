import numpy as np

def relu(x):
    return np.maximum(0.01 * x, x) # Leaky ReLU

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))