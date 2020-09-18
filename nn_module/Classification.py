import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import NeuralNet_class as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist.load_data()
X = np.concatenate([np.array(mnist[0][0], dtype = np.double), np.array(mnist[1][0], dtype = np.double)], axis=0)
X = np.reshape(X, (X.shape[0], X.shape[1]* X.shape[2]))
y_real = np.concatenate([np.array(mnist[0][1]), np.array(mnist[1][1])], axis=0)

X_train, y_train, X_test, y_test = nn.split_train_test(X, y_real, 0.8)
lr = 1e-1

net = nn.Net([nn.Dense(X.shape[1], 1000, "glorot", nn.tanh_activation),
    nn.Dropout(0.4),
    nn.Dense(1000, 500, "glorot", nn.tanh_activation),
    nn.Dropout(0.4),
    nn.Dense(500, 10, "glorot", tf.nn.softmax)],
    nn.categorical_crossentropy)
convert_dict = nn.to_onehot(y_real)
y_train_onehot = np.array([convert_dict[i] for i in y_train])
model = nn.Trainer(net, X_train, y_train_onehot, nn.Sgd(lr, 0.5))
n_epochs = 10
n_batches = 60
model.optimizer.set_decay(n_epochs, lr/10, decay_type="linear")
timestamp = time.time()
tl,vl = model.train(n_epochs, n_batches, learning_rate=lr)
print("Elapsed time:\t"+str(np.round(time.time() - timestamp, 2)))
fig, ax = plt.subplots(1, 2)
ax[0].plot(tl)
ax[1].plot(vl)
ax[0].set_title("Train loss")
ax[1].set_title("Validation loss")
plt.show()

y_pred_onehot = model.net.forward_pass(tf.convert_to_tensor(X_test), test = True)
y_pred = np.argmax(y_pred_onehot, axis = 1)
#y_test = np.argmax(y_test_onehot, axis = 1)

print("Accuracy: " + str(accuracy_score(y_test, y_pred)))

n_classes = y_train_onehot.shape[1]
labels = np.unique(y_real)
cm = confusion_matrix(y_test, y_pred)
normalised_confusion_matrix = np.array(cm, dtype=np.float32)
width = 4
height = 4
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest'
)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, labels, rotation=90)
plt.yticks(tick_marks, labels)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()