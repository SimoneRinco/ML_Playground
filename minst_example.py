import tensorflow as tf
import matplotlib.pyplot as plt


# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

print(f"Shape of training_images is {training_images.shape}, type is {training_images.dtype}")
print(f"Shape of training_labels is {training_labels.shape}, type is {training_labels.dtype}")

# 0 	T-shirt/top
# 1 	Trouser
# 2 	Pullover
# 3 	Dress
# 4 	Coat
# 5 	Sandal
# 6 	Shirt
# 7 	Sneaker
# 8 	Bag
# 9 	Ankle boot


show_stuff = False

if show_stuff:
  all_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"];
  displayed = [False] * len(all_labels)

  for index, label in enumerate(training_labels):
    if all(displayed):
      break
    print(f"label is {all_labels[label]}")
    displayed[label] = True
    plt.imshow(training_images[index],cmap="Greys")
    plt.show()

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential()

# flatten the data from a 28x28 array into a 784x1 array
model.add(tf.keras.layers.Flatten())

# relu (rectified linear unit)
# with default parameters: y = max(x, 0)
# units is the dimensionality of the output space
model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))

# we can add a second dense layer with different size
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))

# units here is 10, as we have 10 labels
# softmax: transforms a vector into a probability distribution (sum = 1, values in [0, 1])
# In this case the component with highest probability is our labeling
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


# optimizer
#   Adam: one of the available algorithms
# loss
#   sparse_categorical_crossentropy: used when there are two or more label classes
#   and the labels to be provided as integers.
# metrics
#   A metric is a function that is used to judge the performance of your model.
#   Metric functions are similar to loss functions, except that the results from evaluating
#   a metric are not used when training the model. Note that you may use any loss function as a metric.
#   Calculates how often predictions equal labels.
#   accuracy
#     This metric creates two local variables, total and count that are used to compute
#     the frequency with which y_pred matches y_true.
#     This frequency is ultimately returned as binary accuracy: an idempotent operation that simply divides total by count.
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


class MyCallback(tf.keras.callbacks.Callback):

  # Called at the end of an epoch
  def on_epoch_end(self, epoch, logs={}):
    # logs contain a lot of info about the training
    min_accuracy = 0.85
    if(logs.get('accuracy') >= min_accuracy):
      print("\nReached {min_accuracy*100}% accuracy so cancelling training!")
      # set this to stop the training
      self.model.stop_training = True

model.fit(training_images, training_labels, epochs=5, callbacks=[MyCallback()])

model.evaluate(test_images, test_labels)