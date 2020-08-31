import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import NeuralNet_class as nn

net = nn.Net(1, [1000, 1], [nn.sigmoid_activation] + [tf.identity], nn.MSE_loss)

X = np.linspace(-1,1, 100).reshape((100, 1))
X = (X-np.mean(X, axis = 0))/np.std(X, axis = 0)
y_real = np.exp(3*X)
lr = 1e-2
model = nn.Trainer(net, X, y_real, nn.Sgd(lr))
tl,vl = model.train(50, 1, learning_rate=lr)
y_pred = model.net.forward_pass(X)
plt.figure()
plt.plot(X, y_pred, "o", label ="pred", markersize = 1)
plt.plot(X, y_real, "o", label ="real", markersize = 1)
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].plot(tl)
ax[1].plot(vl)
ax[0].set_title("Train loss")
ax[1].set_title("Validation loss")
plt.show()
