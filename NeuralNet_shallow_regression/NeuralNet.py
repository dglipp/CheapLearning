import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

class Layer:
    def __init__(self, n_input, n_neurons, activation, 
    init_weights = None, init_bias = None):
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.weights = init_weights
        self.biases = init_bias
        self.activation = activation
        if self.weights is None:
            self.weights = (np.random.randn(n_input, n_neurons))*1
        assert ((n_input, n_neurons) == self.weights.shape)
        if self.biases is None:
            self.biases = np.zeros((1, n_neurons))
        assert ((1, n_neurons) == self.biases.shape)
        self.weights = self.weights
    def forward(self, X):
        return self.activation(X @ self.weights + self.biases).numpy()

class FFNN:
    def __init__(self, n_input, n_layers, n_neurons, activation = None):
        self.n_input = n_input
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.layers = []
        assert len(n_neurons) == n_layers
        if activation is None:
            self.activation = [tf.nn.relu] * (self.n_layers - 1) + [tf.identity]
        else:
            if isinstance(activation, list):
                assert len(activation) == n_layers
                self.activation = activation
            else:
                self.activation = [activation] * (self.n_layers - 1) + [tf.identity]

        temp_neurons = [self.n_input] + self.n_neurons
        for i in range(self.n_layers):
            self.layers.append(Layer(temp_neurons[i], temp_neurons[i+1], self.activation[i]))

    def get_weights(self):
        w = []
        b = []
        for l in self.layers:
            w.append(l.weights)
            b.append(l.biases)
        return w, b
    
    def set_weights(self, weights, biases):
        for i in range(self.n_layers):
            self.layers[i].weights = weights[i]
            self.layers[i].biases = biases[i]

    def forward(self, X, y = None):
        p = X
        for i in range(self.n_layers):
            p = self.layers[i].forward(p)
        if y is not None:
            loss = np.mean((y - p)**2)
            return p, loss
        else:
            return p
    
    def backprop(self, X_batch, y_batch):
        Z = X_batch.copy()
        tW = []
        tb = []
        with tf.GradientTape(persistent = True) as tape:
            for i in range(self.n_layers):
                tW.append(tf.Variable(tf.convert_to_tensor(self.layers[i].weights)))
                tb.append(tf.Variable(tf.convert_to_tensor(self.layers[i].biases)))
                Z = self.layers[i].activation(Z @ tW[i] + tb[i])
            loss = tf.reduce_sum((Z - y_batch)**2)
        dldw = tape.gradient(loss, tW)
        dldb = tape.gradient(loss, tb)
        del tape
        return dldw, dldb
    
    def train(self, X, X_valid, y, y_valid, batch_size, n_epochs, learning_rate = 1e-6, valid_min = True):
        loss = np.empty((n_epochs), dtype=float)
        valid_loss = np.empty((n_epochs), dtype=float)
        prev_dldw = []
        prev_dldb = []
        w_hist = []
        b_hist = []
        for i in range(n_epochs):
            for j in range(int(X.shape[0]/batch_size)):
                momentum = 0
                shuffle = int(np.random.rand()*(X.shape[0] - batch_size))
                y_batch = y[shuffle:(shuffle+1)*batch_size]
                X_batch = X[shuffle:(shuffle+1)*batch_size]
                dldw, dldb = self.backprop(X_batch, y_batch)
                if i == 0 and j == 0:
                    for k in range(self.n_layers):
                        self.layers[k].weights -= learning_rate*(dldw[k].numpy())
                        self.layers[k].biases -= learning_rate*(dldb[k].numpy())
                else:
                    for k in range(self.n_layers):
                        self.layers[k].weights -= learning_rate*(dldw[k].numpy() - momentum * prev_dldw[k].numpy())
                        self.layers[k].biases -= learning_rate*(dldb[k].numpy() - momentum * prev_dldb[k].numpy())
            
                prev_dldw = dldw
                prev_dldb = dldb
            print(dldw[0].numpy())
            print("\n")
            #print(dldb[0].numpy())
            print("\n")
            hw, hb = self.get_weights()
            w_hist.append(hw)
            b_hist.append(hb)
            _, loss[i] = self.forward(X, y)
            _, valid_loss[i] = self.forward(X_valid, y_valid)
            print("                                                                                ", end = "\r")
            print("epoch: " + str(i) + 
            "\tloss: " + str(np.round(loss[i], 3)) +
            "\t\tvalid_loss: " + str(np.round(valid_loss[i], 3)), end = "\r")
        print("\n")
        if valid_min == True:
            c = np.argmin(valid_loss)
            self.set_weights(w_hist[c], b_hist[c])
        return w_hist, b_hist, loss, valid_loss
    

n_feats = 2
n_neurons = [100, 1]
n_batches = 1
batch_size = 10
n_epochs = 1000
valid_size = 50
lr = 1e-5

net = FFNN(n_feats, len(n_neurons), n_neurons, activation = tf.nn.relu)

X = np.random.randn(batch_size*n_batches, n_feats)
X_sort = np.sort(X, axis=0)
X_valid = np.random.randn(valid_size, n_feats)
X_valid_sort = np.sort(X, axis=0)
y = np.sum(np.sin(6*X), axis = 1)
y_sort = np.sum(np.sin(6*X_sort), axis = 1)
y_valid = np.sum(np.sin(6*X_valid), axis = 1)
y_valid_sort = np.sum(np.sin(6*X_valid_sort), axis = 1)



historiaw, historiab, loss, valid_loss = net.train(X, X_valid, y, 
y_valid, batch_size, n_epochs, learning_rate=lr, valid_min=False)

fig, ax = plt.subplots(1, 2)
ax[0].plot(loss)
ax[0].set_title("train loss")
ax[1].plot(valid_loss)
ax[1].set_title("validation loss")
plt.show()

X_test = np.sort(np.random.randn(100, n_feats), axis=0)
y_test = net.forward(X_test)

fig, ax = plt.subplots(n_feats, 1)

for i in range(n_feats):
    ax[i].plot(X_test[:,i], y_test, "-", color = "black", label = "test set")
    ax[i].plot(X_sort[:,i], y_sort, "-", color = "blue", label = "train set")
    ax[i].plot(X_valid_sort[:,i], y_valid_sort, "-", color = "red", label = "validation set")
plt.legend()
plt.show()
