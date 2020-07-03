import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np_w = np.array([[1,2],[2,1],[3,3]], dtype=float)
w = tf.Variable(tf.convert_to_tensor(np_w), name='w')
np_b = np.array([20,-20], dtype=float)
b = tf.Variable(tf.convert_to_tensor(np_b), name='b')
x = np.array([1., 2., 3.]).reshape((1, -1))

@tf.custom_gradient
def custom_sigmoid(y):
    def backward(dy):
        df = dy*tf.sigmoid(y)*(1-tf.sigmoid(y))
        return df
    return tf.sigmoid(y), backward

@tf.custom_gradient
def custom_relu(y):
    def backward(dy):
        df = tf.cast(y > tf.cast(tf.zeros(y.shape), dtype=y.dtype), dtype = y.dtype) * dy
        return df
    return tf.nn.relu(y), backward

with tf.GradientTape(persistent=True) as tape:
    y1 = tf.nn.relu(x @ w + b)
    y2 = custom_relu(x @ w + b)

print(x @ w + b)
print(y2)
print(tape.gradient(y1, w))
print(tape.gradient(y2, w))