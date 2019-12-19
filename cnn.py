"""
Convolutional neural network architecture.
Author: Yongxin (Fiona) Xu + Amberley Su
Date: 12/18/2019
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class CNNmodel(Model):
    """
    A convolutional neural network; the architecture is:
    Conv -> ReLU -> Conv -> ReLU -> Dense
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(CNNmodel, self).__init__()
        # First conv layer: 32 filters, each 5x5
        self.conv1 = Conv2D(32, 5, activation="relu")
        # Second conv layer: 16 filters, each 3x3
        self.conv2 = Conv2D(16, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.d1(x)