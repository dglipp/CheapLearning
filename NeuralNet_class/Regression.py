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
tl,vl = model.train(3, 1, learning_rate=lr)
y_pred = model.net.forward_pass(X)
plt.figure()
plt.plot(X, y_pred, "o", label ="pred", markersize = 1)
plt.plot(X, y_real, "o", label ="real", markersize = 1)
plt.legend()
plt.show()
plt.figure()
plt.plot(tl, label="train_loss")
plt.plot(vl, label="validate_loss")
plt.legend()
plt.show()

X = tf.convert_to_tensor(np.random.randn(3, 4))
w = tf.Variable(tf.convert_to_tensor(np.random.randn(2, 3)))


with tf.GradientTape(persistent=True) as tape:
    y1 = tf.nn.softmax(w @ X)
    #y2 = nn.softmax(w @ X)
print(y1)
print(tape.gradient(y1, w))
#print(tape.gradient(y2, w))