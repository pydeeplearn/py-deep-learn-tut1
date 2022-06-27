#!/usr/bin/evn python
# python 3.8, tensorflow 2.2.0
# https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf

# load and prepare the MNIST dataset (http://yann.lecun.com/exdb/mnist/).
# Convert the samples from integers to floating-point numbers:

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers.
# Choose an optimizer and loss function for training:

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# For each example the model returns a vector of "logits" or "log-odds" scores,
# one for each class.

predictions = model(x_train[:1]).numpy()
print("predictions\n", predictions)

# The tf.nn.softmax function converts these logits to "probabilities" for each class:

probabilities = tf.nn.softmax(predictions).numpy()
print("probabilityes\n", probabilities)

