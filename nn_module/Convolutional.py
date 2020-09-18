import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import NeuralNet_class as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
X = np.random.randn(2, 3, 10,10)
net = nn.Net([nn.Convolutional(3, 4, (3,3), 1),
            nn.Pooling((3,3))], nn.categorical_crossentropy)

