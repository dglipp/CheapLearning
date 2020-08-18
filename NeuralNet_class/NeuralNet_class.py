import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#define loss functions
@tf.custom_gradient
def MSE_loss(y_pred, y_real):
    def backward(dy):
        df = np.reshape(dy, y_pred.shape) * (y_pred - y_real)*2/y_pred.shape[1]
        return df, df
    return tf.reduce_mean((y_pred - y_real)**2, axis=1), backward

#define activation functions
@tf.custom_gradient
def sigmoid_activation(X):
    s = 1/(1 + tf.exp(-X))
    def backward(dy):
        df = dy * s * (1-s)
        return df
    return s, backward

#define optimizer class
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.net = net
    
    def update(self, weights, derivatives):
        pass
    
class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update(self, parameters, derivatives):
        for i, p in enumerate(parameters):
            p.assign_sub(self.learning_rate * derivatives[i])

#define layer class
class Layer:
    def __init__(self, n_inputs, n_neurons, activation = None):
        if activation == None:
            self.activation = tf.identity
        else:
            self.activation = activation
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.W = tf.Variable(tf.convert_to_tensor(np.random.randn(n_inputs, n_neurons)))
        self.b = tf.Variable(tf.convert_to_tensor(np.zeros((1, n_neurons))))

    def forward(self, _X):
        X = tf.convert_to_tensor(_X)
        return self.activation(X @ self.W + self.b)

#define neural net class
class Net:
    def __init__(self, n_inputs, layers_neurons, activations, loss):
        self.numbers = [n_inputs] + layers_neurons
        self.activations = activations
        self.layers = list()
        self.loss = loss
        for i in range(len(layers_neurons)):
            self.layers.append(Layer(self.numbers[i], self.numbers[i+1], activation = activations[i]))
    
    def forward_pass(self, _X):
        y = tf.convert_to_tensor(_X)
        for l in self.layers:
            y = l.forward(y)
        return y

    def get_loss(self, _X, _y_real):
        X = tf.convert_to_tensor(_X)
        y_real = tf.convert_to_tensor(_y_real)
        y_pred = self.forward_pass(X)
        return tf.reduce_mean(self.loss(y_pred, y_real))

    def get_weights(self):
        weights = list()
        for l in self.layers:
            weights.append(l.W)
            weights.append(l.b)
        return weights

#define training class
class Trainer():
    def __init__(self, net, _X, _y, optimizer, shuffle = True):
        self.X = _X
        self.y = _y
        if shuffle:
            self.shuffle_data(self.X, self.y)
        self.net = net
        self.optimizer = optimizer
    
    def split_train_validate(self, train_percentage):
        X_train = self.X[:int(self.X.shape[0]*train_percentage)]
        X_validate = self.X[int(self.X.shape[0]*train_percentage):]
        y_train = self.y[:int(self.y.shape[0]*train_percentage)]
        y_validate = self.y[int(self.y.shape[0]*train_percentage):]
        return X_train, y_train, X_validate, y_validate
    
    def shuffle_data(self, _X, _y):
        rng_state = np.random.get_state()
        np.random.shuffle(_X)
        np.random.set_state(rng_state)
        np.random.shuffle(_y)
    
    def create_batches(self, _X, _y, n_batches):
        batch_size = int(_X.shape[0]/n_batches)
        X_batches = list()
        y_batches = list()
        if n_batches>1:
            for i in range(n_batches - 1):
                X_batches.append(_X[batch_size*i:batch_size*(i+1)])
                y_batches.append(_y[batch_size*i:batch_size*(i+1)])
            X_batches.append(_X[batch_size*i:])
            y_batches.append(_y[batch_size*i:])
        else:
            X_batches.append(_X)
            y_batches.append(_y)
        return X_batches, y_batches
    
    def train(self, n_epochs, n_batches,  learning_rate=0.01, train_validate_percentage = 0.7, shuffle = True):
        weights = self.net.get_weights()
        X_train, y_train, X_validate, y_validate = self.split_train_validate(train_validate_percentage)
        train_loss = list()
        validate_loss = list()
        for n in range(n_epochs):
            if shuffle:
                self.shuffle_data(X_train, y_train)
            X_batches, y_batches = self.create_batches(X_train, y_train, n_batches)
            for i in range(n_batches):
                with tf.GradientTape() as tape:
                    loss = net.get_loss(X_batches[i], y_batches[i])
                train_loss.append(loss.numpy())
                print("                                                                                ", end="\r")
                print("epoch: " + str(n) + 
                "\ttrain loss: " + str(np.round(loss.numpy(), 3)), end="\r")
                grad = tape.gradient(loss, weights)
                self.optimizer.update(weights, grad)
            validate_loss.append(net.get_loss(X_validate, y_validate).numpy())
            print("                                                                                ", end="\r")
            print("epoch: " + str(n) + 
            "\ttrain loss: " + str(np.round(loss.numpy(), 3)) + 
            "\tvalidation loss: " + str(np.round(validate_loss[-1], 3)), end="\r")
            print("\n")
        return train_loss, validate_loss

net = Net(1, [1000, 1], [sigmoid_activation] + [tf.identity], MSE_loss)

X = np.linspace(-1,1, 100).reshape((100, 1))
X = (X-np.mean(X, axis = 0))/np.std(X, axis = 0)
y_real = np.exp(3*X)

lr = 1e-2
model = Trainer(net, X, y_real, Sgd(lr))
tl,vl = model.train(30, 1, learning_rate=lr)
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


