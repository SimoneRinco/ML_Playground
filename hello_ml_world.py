import tensorflow as tf
import numpy as np
from tensorflow import keras


# Creates a model formed by a linear sequence of layers
model = tf.keras.Sequential()

# Add one layer (layers are the basic building blocks of neural networks in Keras)
model.add(keras.layers.Dense(units=1, input_shape=[1]))

# Configure the model for training
model.compile(optimizer='sgd', loss='mean_squared_error')

# Fit (train) the model
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)

# Predict a bunch of values
print(model.predict([0.0, 10.0, 100.0]))
