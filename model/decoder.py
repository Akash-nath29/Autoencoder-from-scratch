from functions.utils import dense, init_params
from functions.activations import sigmoid
import numpy as np

class Decoder:
    def __init__(self, hidden_size, output_size, optimizer):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.W2, self.b2 = init_params(hidden_size, output_size)
        self.m_w2 = self.v_w2 = np.zeros_like(self.W2)
        self.m_b2 = self.v_b2 = np.zeros_like(self.b2)

    def forward(self, x):
        self.x = x
        self.z2 = dense(x, self.W2.T, self.b2.T)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, y):
        m = y.shape[0]
        self.dz2 = self.a2 - y
        self.dW2 = np.dot(self.dz2.T, self.x) / m
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True).T / m

    def update(self):
        self.W2, self.m_w2, self.v_w2 = self.optimizer.update(self.W2, self.dW2, self.m_w2, self.v_w2)
        self.b2, self.m_b2, self.v_b2 = self.optimizer.update(self.b2, self.db2, self.m_b2, self.v_b2)