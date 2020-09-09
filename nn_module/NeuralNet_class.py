import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#define loss functions
@tf.custom_gradient
def MSE_loss(y_real, y_pred):
    def backward(dy):
        df = np.reshape(dy, y_pred.shape) * (y_pred - y_real)*2/y_pred.shape[1]
        return df, df
    return tf.reduce_mean((y_pred - y_real)**2, axis=1), backward

@tf.custom_gradient
def categorical_crossentropy(y_real, y_pred):
    loss = tf.reduce_sum(- y_real * tf.math.log(y_pred)  - (1-y_real) * tf.math.log(1 - y_pred), axis = 1)
    def backward(dy):
        df = np.reshape(dy, (dy.shape[0], 1))*(- y_real/(y_pred) + (1-y_real)/(1-y_pred))
        return df, df
    return loss, backward

#define activation functions
@tf.custom_gradient
def sigmoid_activation(X):
    s = 1/(1 + tf.exp(-X))
    def backward(dy):
        df = dy * s * (1-s)
        return df
    return s, backward

@tf.custom_gradient
def relu_activation(X):
    def backward(dy):
        return dy*np.greater(X, 0).astype(np.float64)
    return np.maximum(0,X), backward

@tf.custom_gradient
def tanh_activation(X):
    t = tf.tanh(X)
    def backward(dy):
        return dy*(1-t**2)
    return t, backward

#util functions
def to_onehot(labels):
    labs = list(np.unique(labels))
    onehot = np.diag(np.ones(len(labs))).tolist()
    return dict(zip(labs, onehot))

def split_train_test(_X, _y, train_percentage):
    X = _X.copy()
    y = _y.copy()
    shuffle_data(X, y)
    X_train = X[:int(X.shape[0]*train_percentage)]
    X_test = X[int(X.shape[0]*train_percentage):]
    y_train = y[:int(y.shape[0]*train_percentage)]
    y_test = y[int(y.shape[0]*train_percentage):]
    return X_train, y_train, X_test, y_test
    
def shuffle_data(_X, _y):
    rng_state = np.random.get_state()
    np.random.shuffle(_X)
    np.random.set_state(rng_state)
    np.random.shuffle(_y)

#define optimizer class
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weights, derivatives):
        pass

class Sgd(Optimizer):
    def __init__(self, learning_rate, momentum=0):
        super().__init__(learning_rate)
        self.momentum=momentum
        self.prev_grad = []
        self.first=True
        self.decay_type = None
        self.delta_lr = 0

    def set_decay(self, n_epochs, final_lr, decay_type="linear"):
        self.decay_type=decay_type
        if decay_type == "linear":
            self.delta_lr = (self.learning_rate - final_lr)/(n_epochs - 1)
        if decay_type == "exponential":
            self.decay_type = np.power(final_lr/self.learning_rate, 1/(n_epochs -1))

    def update_decay(self):
        if self.decay_type == "linear":
            self.learning_rate -= self.delta_lr
        if self.decay_type == "exponential":
            self.learning_rate *= self.delta_lr

    def update(self, parameters, derivatives):
        for i, p in enumerate(parameters):
            if self.first is True:
                p.assign_sub(self.learning_rate * derivatives[i])
                self.prev_grad.append(derivatives[i])
            else:
                p.assign_sub(self.learning_rate *( derivatives[i] + self.momentum * self.prev_grad[i]))
                self.prev_grad[i] = derivatives[i] + self.momentum * self.prev_grad[i]
        self.first = False
        
#define layer class
class Layer:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs

    def forward(self):
        pass

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, w_init, activation = None):
        super().__init__(n_inputs)
        if activation == None:
            self.activation = tf.identity
        else:
            self.activation = activation
        self.n_neurons = n_neurons
        if w_init is None:
            s = 1.0
        if w_init == "glorot":
            s = 2/(n_inputs + n_neurons)
        self.W = tf.Variable(tf.convert_to_tensor(np.random.randn(n_inputs, n_neurons)*np.sqrt(s)))
        self.b = tf.Variable(tf.convert_to_tensor(np.zeros((1, n_neurons))))

    def forward(self, _X):
        X = tf.convert_to_tensor(_X)
        return self.activation(X @ self.W + self.b)

class Dropout(Layer):
    def __init__(self, freq):
        super().__init__(None)
        self.freq = freq
    
    def forward(self, _X):
        X = tf.convert_to_tensor(_X)
        return X * np.random.binomial(1, 1-self.freq, size = X.shape)

#define neural net class
class Net:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.l_types = (Dense)
    
    def forward_pass(self, _X, test = False):
        y = tf.convert_to_tensor(_X)
        if test:
            for l in self.layers:
                if isinstance(l, self.l_types):
                    y = l.forward(y)
        else:
            for l in self.layers:
                y = l.forward(y)
        return y

    def get_loss(self, _X, _y_real, test = False):
        X = tf.convert_to_tensor(_X)
        y_real = tf.convert_to_tensor(_y_real)
        y_pred = self.forward_pass(X, test)
        return tf.reduce_mean(self.loss(y_real, y_pred))

    def get_weights(self):
        weights = list()
        for l in self.layers:
            if isinstance(l, self.l_types):
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
                    loss = self.net.get_loss(X_batches[i], y_batches[i])
                train_loss.append(loss.numpy())
                print("                                                                                ", end="\r")
                print("epoch: " + str(n) + 
                "\ttrain loss: " + str(np.round(loss.numpy(), 3)), end="\r")
                grad = tape.gradient(loss, weights)
                self.optimizer.update(weights, grad)
            validate_loss.append(self.net.get_loss(X_validate, y_validate, test = True).numpy())
            print("                                                                                ", end="\r")
            print("epoch: " + str(n) + 
            "\ttrain loss: " + str(np.round(loss.numpy(), 3)) + 
            "\tvalidation loss: " + str(np.round(validate_loss[-1], 3)), end="\r")
            print("\n")
            self.optimizer.update_decay()
        return train_loss, validate_loss
