import numpy as np 

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

l1 = Layer(4, 6)
l2 = Layer(6, 2)

X = np.random.randn(3,4)
l1.forward(X)
l2.forward(l1.output)
print(l2.output)