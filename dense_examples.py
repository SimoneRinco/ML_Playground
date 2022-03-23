from keras import layers, activations
import numpy as np
import tensorflow as tf

def example_1():

  # This example shows that a Dense layer simply performs the operation
  # output = activation(input . weights + bias)
  # input is a tensor, weights is a matrix and bias is a vector (the + operation
  # will trigger broadcasting.

  # Example
  # input: (3 x 4)
  # |  3 -1  6  4 |
  # | -2  1  0  5 |
  # | 12 -3 -5  0 |
  #
  # weights (also called kernel): (4 x 2)
  # | -4  1 |
  # |  1 -6 |
  # | -1  0 |
  # |  2  3 |
  #
  # bias: can be thought as (1 x 2)
  # | 12  -3|
  #
  # input . weights + bias: (3 x 2)
  #
  # |   1  18 |
  # |  31   4 |
  # | -34  27 |
  #
  # This is simply a linear transformation. To introduce a non linearity, we apply an activation
  # function. For example a relu with a threshold of 10 will produce
  #
  # |  0  18 |
  # | 31   0 |
  # |  0  27 |
  #
  # Interpretation
  # The input represents 3 samples and each sample is described with 4 numbers.
  # The Dense layer will transform each sample from a 4 dimensional space to a 2 dimensional space
  # (the number of units is the Dense layer).
  # For example the first sample [3 -1 6 4] is transformed into
  # | relu(3*-4 + -1* 1 + 6*-1 + 4*2 + 12) | = |  0 |
  # | relu(3* 1 + -1*-6 + 6* 0 + 4*3 + -3) |   | 18 |
  #
  # Let's see all of this in action

  # Size of the output space.
  units = 2

  def my_bias_initializer(shape, dtype=None):
    assert len(shape) == 1
    assert shape[0] == 2
    return tf.constant([12, -3], dtype=dtype)

  def my_kernel_initializer(shape, dtype=None):
    assert len(shape) == 2
    assert shape[0] == 4
    assert shape[1] == 2
    return tf.constant([[-4, 1], [1, -6], [-1, 0], [2, 3]], dtype=dtype)

  def relu_wrapper():
    def do_relu(x):
      return activations.relu(x, threshold=10)
    return do_relu

  # We need to pass only the dimension of the output space (units). The number of features and
  # number of samples is not known yet.
  # This means kernel and bias are NOT constructed at this point.
  l = layers.Dense(units=units,
                   use_bias=True,
                   bias_initializer=my_bias_initializer,
                   kernel_initializer=my_kernel_initializer,
                   activation=relu_wrapper())

  # Kernel and bias are initialised calling build, where we pass the shape of the input (number of
  # features of each sample).
  l.build(input_shape=(4))

  print("=== W:\n", l.weights[0])
  print("=== b:\n", l.weights[1])

  # |  3 -1  6  4 |
  # | -2  1  0  5 |
  # | 12 -3 -5  0 |

  input_tensor = tf.constant([[3, -1, 6, 4], [-2, 1, 0, 5], [12, -3, -5, 0]], dtype=np.float32)
  output_tensor = l(input_tensor)
  print(output_tensor)

if __name__ == "__main__":
  example_1()
