from functions.activations import relu
from functions.utils import dense, init_params
import numpy as np

class Encoder:
    def __init__(self, input_size, hidden_size, optimizer):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.optimizer = optimizer
        self.W1, self.b1 = init_params(input_size, hidden_size)
        self.m_w1 = self.v_w1 = np.zeros_like(self.W1)
        self.m_b1 = self.v_b1 = np.zeros_like(self.b1)

    def forward(self, x):
        self.x = x
        self.z1 = dense(x, self.W1.T, self.b1.T)
        self.a1 = relu(self.z1)
        return self.a1
    
    def backward(self, y):
        m = y.shape[0]
        dz1_raw = np.dot(self.W1, self.x.T)
        self.dz1 = dz1_raw.T * (self.a1 > 0)
        self.dW1 = np.dot(self.dz1.T, self.x) / m
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True).T / m

    def update(self):
        self.W1, self.m_w1, self.v_w1 = self.optimizer.update(self.W1, self.dW1, self.m_w1, self.v_w1)
        self.b1, self.m_b1, self.v_b1 = self.optimizer.update(self.b1, self.db1, self.m_b1, self.v_b1)