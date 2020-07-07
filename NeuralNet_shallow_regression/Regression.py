import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

n_neurons = 1000
n_output = 1
batch_size = 100
n_feats = 1
n_epochs = 10000
lr = 1e-5
momentum= 0.9

X = np.random.rand(batch_size, n_feats)*5-2.5
y = np.sin(3*X)
W1 = np.random.randn(n_feats, n_neurons)
b1 = np.zeros((1, n_neurons))
tW1 = tf.Variable(tf.convert_to_tensor(W1))
tb1 = tf.Variable(tf.convert_to_tensor(b1))
W2 = np.random.randn(n_neurons, n_output)
b2 = np.zeros((1, n_output))
tW2 = tf.Variable(tf.convert_to_tensor(W2))
tb2 = tf.Variable(tf.convert_to_tensor(b2))

tot_w = [tW1, tb1,  tW2, tb2]


X_test = np.asmatrix(np.linspace(-5,5, 100)).T
fig, ax = plt.subplots(1,2)
prev_dl = []
for i in range(n_epochs):

    with tf.GradientTape() as tape:
        P = tf.nn.sigmoid(X @ tot_w[0] + tot_w[1]) @ tot_w[2] + tot_w[3]
        loss = tf.reduce_sum((P - y)**2)
    dl = tape.gradient(loss, tot_w)
    for j in range(len(tot_w)):
        if i == 0 and j == 0:
            tot_w[j].assign_sub(lr * (dl[j]))
        else:
            tot_w[j].assign_sub(lr * (dl[j] + momentum*prev_dl[j]))
        prev_dl = dl
    print("                                                                                ", end = "\r")
    print("epoch: " + str(i) + 
        "\tloss: " + str(np.round(loss.numpy(), 3)), end = "\r")
print("\n")

y_test = (tf.nn.sigmoid(X_test @ tot_w[0] + tot_w[1]) @ tot_w[2] + tot_w[3]).numpy()
ax[0].plot(X_test, y_test, "-", color = "blue")
ax[0].plot(X, y, "o", markersize = 1, color = "red")

plt.show()


