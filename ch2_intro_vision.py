import tensorflow as tf
import numpy as np
import tensorflow.python.keras as keras
from keras import Sequential
from keras.layers import Flatten, Dense

data = tf.keras.datasets.fashion_mnist

(training_imgs, training_labels), (test_imgs, test_labels) = data.load_data()

training_imgs = training_imgs / 255
test_imgs = test_imgs / 255

''' --- Keywords ---
Hyperparameter
    - these are the values that determine how the training is controlled
    - parameters are known to be the internal value of the neurons that a model uses to train/learn.
'''


''' model init explanation
Sequential - used to define layers, inside of it you define what those layers will be (what they will look like)
Flatten - is not a layer of neurons, but an "input layer specification."
            - our inputs are 28 x 28 images, but "we want to treat them as a series of numeric values." Move from 2D array => 1D Array
            - Flatten takes the square value of the 2D array and returns a 1D Array (a line)
Dense   - used to define the number of neurons we want in a layer
            - we use 128 (in the first one), and that is an arbitrary number
            - the more neurons means the model has to learn more parameters, thus causing the model to run slower
            - by having more neurons, a model can become great at recognizing training data, but can be bad at recognizing unrecognized data
                - called 'overfitting' more on this later
            - with too little neurons, the model may have a difficult time learning effectively due to having too little parameters
            - Thought:
                - is the ideal to go from a high number of neurons to train recognized data, but then to reduce the number as you input more unrecognized data? Is there a correlation between number of neurons and a models ability to recognize previously unseen data?
        - this first Dense method is described as the 'middle layer'
            - this is the layer between the inputs and outputs
            - also known as the 'hidden layer' since this layer is not seen by the caller
        - relu
            - essentially, will only return positive numbers
            - this activation function will be called on each neuron (128 * number of epochs)
Dense   - again, used to define a layer of neurons
        - 10 neurons
            - only 10 neurons because we have 10 classes (10 different clothing types)
            - each of the 10 neurons will be given a number determining the probability that the input pixels match that class
        - softmax
            - this will determine which neuron (out of the 10) has the largest value
'''
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(training_imgs, test_imgs, epochs=5)