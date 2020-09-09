import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import NeuralNet_class as nn

net = nn.Net([nn.Dense(1, 100, "glorot", nn.sigmoid_activation),
    nn.Dense(100, 50, "glorot", nn.sigmoid_activation),
    nn.Dense(50, 1, "glorot", tf.identity)],
    nn.MSE_loss)
X = np.linspace(-1,1, 100).reshape((100, 1))
X = (X-np.mean(X, axis = 0))/np.std(X, axis = 0)
y_real = np.exp(3*X)
lr = 1e-2
n_epochs = 500
model = nn.Trainer(net, X, y_real, nn.Sgd(lr))
model.optimizer.set_decay(n_epochs, lr/5000, decay_type="exponential")
timestamp = time.time()
tl,vl = model.train(n_epochs, 1, learning_rate=lr)
print("Elapsed time:\t"+str(np.round(time.time() - timestamp, 2)))
y_pred = model.net.forward_pass(X, test=True)
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
