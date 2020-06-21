import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Derivative:
    def __init__(self, x, chain):
        self.x = x
        self.chain = chain
    
    def value(self, input = None):
        if input is None:
            input = self.x
        y = np.copy(input)
        for f in self.chain:
            y = f(y)
        return y

    def derivative(self, func, input, delta: float = 0.001):
        return (func(input + delta) - func(input - delta))/(2*delta)
    
    def compDerivative(self, input = None):
        if input is None:
            input = self.x
        d = np.ones(len(input))
        f = np.empty(shape=[len(self.chain)+1, len(input)])
        f[0] = input
        for i in range(len(chain)):
            f[i+1] = chain[i](f[i])
            d = self.derivative(chain[i], f[i])*d
        return d


def square(x):
    return x**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-3, 3, 0.01)
chain = [sigmoid, square]
d = Derivative(x, chain)

plt.plot(x, d.value(), color=sns.color_palette()[0], label = "f", linewidth=0.8)
plt.plot(x, d.compDerivative() , color = sns.color_palette()[1], label = "df/dx", linewidth=0.8)
plt.legend()
plt.show()
