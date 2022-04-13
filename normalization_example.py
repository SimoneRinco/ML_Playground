import tensorflow as tf
from tensorflow import keras
from keras import Input, layers
from keras.models import Model

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import sys
import math

# This script shows how inputs and output can be normalized in order to
# improve convergence of the NN model.
# Suppose we have a regression model where
#   x = [-alpha, alpha]   input
# and
#   y = beta * (x - 1)    output
#
# When x and/or y are "far from 1" (alpha/beta very large or very small)
# the NN might struggle to converge.
# Normalizing the data means transforming input and output
# such that the data is distributed "around 0 and not too big", such
# as having mean 0 and variance 1.
# To achieve that, both input and output variables in the training set
# are linearly transformed:
#   xn = (x - mean_x) / sqrt(variance_x)
#   yn = (y - mean_y) / sqrt(variance_y)
#
# Observe that both mean and variance depend on the training set and these
# parameters are NOT trainable.
# The challenge is to keep both input and output of the total model


def get_y(x, beta):
    return beta * (x - 1.0)


def print_info(v, v_norm):
    print("  original vector (first elements)")
    print(np.array(v[:10]))
    print("  normalized vector (first elements)")
    print(np.array(v_norm[:10]))
    print("  normalized info")
    # These must be exactly 0 and 1 respectively!
    print(f"  mean: {np.mean(v_norm)}, variance: {np.var(v_norm)}")


def main(alpha, beta, normalize_input, normalize_output):

    n_samples = 10000

    # Input of the NN
    initial_input = Input(shape=(1, ), name='initial_input')

    x = np.linspace(start=-alpha, stop=alpha, num=n_samples)
    y = get_y(x, beta)

    if normalize_input:
        x_norm_layer = layers.Normalization(axis=None,
                                            name='my_input_normalization')
        # Configure mean and variance based on the input values in the training set
        x_norm_layer.adapt(x)
        x_norm = x_norm_layer(x)
        print("===== Input is going to be normalized")
        print_info(x, x_norm)
        my_input = x_norm_layer(initial_input)
    else:
        my_input = initial_input

    # Two inner layers
    layer1 = layers.Dense(32, activation='linear', name='my_1st_dense')
    # If the input of the first layer is normalized, the values should be
    # small during both training and prediction, if the predicted input is
    # similar (same order of magnitude) to the training data
    out_layer1 = layer1(my_input)

    layer2 = layers.Dense(1, activation='linear', name='my_2nd_dense')
    out_layer2 = layer2(out_layer1)

    if normalize_output:
        y_norm_layer = layers.Normalization(axis=None)
        y_norm_layer.adapt(y)
        y_norm = y_norm_layer(y)
        print("===== Output is going to be normalized")
        print_info(y, y_norm)

        # We want the output of the last dense layer to be normalized
        # and the output of the entire NN to be not normalized. To do so,
        # the last layer should perform the denormalization. Observe that the
        # direct transformation is
        # x' = (x - mean) / sqrt(var)
        # so the inverse is
        # x = (x' - (-mean/sqrt(var)) / (1 / sqrt(var))

        denorm_mean = float(-y_norm_layer.mean /
                            math.sqrt(y_norm_layer.variance))
        denorm_variance = float(1.0 / y_norm_layer.variance)
        print(
            f"Denormalization mean and var: {denorm_mean}, {denorm_variance}")
        y_denorm_layer = layers.Normalization(axis=None,
                                              mean=denorm_mean,
                                              variance=denorm_variance,
                                              name='my_output_denormalization')

        y_denorm = y_denorm_layer(y_norm)
        print("===== Denormalized info")
        # Here y_denorm should be equal to the original y
        print_info(y_denorm, y_norm)
        assert math.fabs(
            y[0] - y_denorm[0]
        ) < 1e-6, f"y[0] and y_denorm[0] are different ({y[0]} vs {y_denorm[0]})"

        # Now we expect the output of the last dense layer to contain small values
        denormalized_output = y_denorm_layer(out_layer2)
        output = denormalized_output
    else:
        output = out_layer2

    # The input of the model is ALWAYS the original (un-normalized) input
    model = Model(initial_input, output)

    model.summary()

    # If the output is denormalized, the loss could be calculated on very
    # big or very small values. For this reason we calculate the mse on
    # normalized true and predicted values.
    def custom_normalised_loss(y_true, y_pred):
        y_true_norm = y_norm_layer(y_true)
        y_pred_norm = y_norm_layer(y_pred)
        squared_difference = tf.square(y_true_norm - y_pred_norm)
        return tf.reduce_mean(squared_difference, axis=-1)

    loss = custom_normalised_loss if normalize_output else 'mse'

    model.compile(optimizer='rmsprop', loss=loss)

    # The NN is fitted with the true, un-normalized data!!!
    model.fit(x=x, y=y, epochs=10)

    # prediction
    x_new = np.linspace(start=-0.2 * alpha, stop=0.2 * alpha, num=5)

    print("===== Prediction")
    print("x_new")
    print(x_new)
    print("y_new_real")
    y_new_real = get_y(x_new, beta)
    print(y_new_real)
    y_new_pred = model.predict(x_new)
    print("y_new_pred")
    print(y_new_pred)

    # No need to apply normalization to the output of the NN!!!
    plt.plot(x_new, y_new_real, color='blue', marker='o')
    plt.plot(x_new, y_new_pred, color='red', marker='o')

    plt.show()


if __name__ == "__main__":
    args = sys.argv
    if (len(args) != 5):
        print("Usage: specify alpha beta normalize_input normalize_output")
        sys.exit(1)

    alpha = float(args[1])
    beta = float(args[2])
    normalize_input = True if int(args[3]) else False
    normalize_output = True if int(args[4]) else False

    main(alpha, beta, normalize_input, normalize_output)

    # Examples (converge result might change due to random initialization of the weights)
    # ========
    # alpha    beta      norm_input    norm_output    converges   comment
    #     1    1.0               no             no          yes   data is already scaled properly, no need for normalization
    # ----
    # 10000    0.0001            no             no           no   input and output are not scaled properly
    # 10000    0.0001           yes             no          yes   with normalization of the input, the case converges
    # ----
    #     1    0.000001          no             no           no   the output is not scaled properly
    #     1    0.000001          no            yes          yes   with normalization of the input, the case converges
    # ----
    #  1000    0.000001          no             no           no
    #  1000    0.000001         yes             no           no
    #  1000    0.000001          no            yes           no
    #  1000    0.000001         yes            yes           no  Both scaling of the input and the output are needed to obtain convergence!
