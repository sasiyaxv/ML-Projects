# -*- coding: utf-8 -*-
"""firstMl

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uoVpYOekbUJI2ZBCWGo6WIs0js1IDYpq
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

print("Tensorflow version is "+tf.__version__)
print("Numpy version is "+np.__version__)

#this example demonstrate( y = 5x - 2) equation
#simplest network possible with only 1 neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


model.compile(optimizer='sgd', loss='mean_squared_error')

#providing data to the model. with x and y
xs = np.array([-1.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([-7.0,3.0, 8.0, 13.0,18.0, 23.0, 28.0], dtype=float)

#fitting data x with y with 400 iterations(epochs)
#in this scenario when the iterations are increased accuracy level also increases

model.fit(xs, ys, epochs=800)

#testing model. 98 is the expected value
print(model.predict([20.0]))