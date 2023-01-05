import tensorflow as tf
import numpy as np
# make sure to either use tensorflow.python.karas or karas, mixing the two causes errors
# import tensorflow.python.keras as keras
from tensorflow.python.keras import Sequential, layers

''' --- Keywords ---
Hyperparameter
    - these are the values that determine how the training is controlled
    - parameters are known to be the internal value of the neurons that a model uses to train/learn.
'''

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochNum, logs={}):
        if (logs.get('accuracy') >= 0.95):
            print(f'\n Reach 95% accuracy so ending training')
            self.model.stop_training = True

callbacks = myCallback()
data = tf.keras.datasets.fashion_mnist

(training_imgs, training_labels), (test_imgs, test_labels) = data.load_data()

''' normalzing the image
In python, this notation allows you to do an operation across the whole array
    - all our images are greyscale, with a value between 0 and 255
    - thus, by dividing by 255, we are making each pixel's value be a number between or at 0 - 255
The overarching math behind why one would do this is not the scope of this book, however
    - know it improves the performance of the model
'''
training_imgs = training_imgs / 255
test_imgs = test_imgs / 255


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
                - the model becomes too dependent on the details and noise of its training data
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
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

''' compile
optimizer
    - adam
        - essentially a better, more efficient version of sgd "stochastic gradient descent"
loss
    - sparse_categorical_crossentropy
        - since we our articles of clothing with be labeled in categories between 0 - 9
        - using a `categorical` loss function is the way to go
    - deciding which loss function is use in any given model is an art in itself
        - as time goes on and you build out more models, you'll get a grasp of when to use which loss function
'''
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

''' model.fit
if we were to increase the epochs from 5 => 50, the model's accuracy increase from rougly 89% to 96%
    - however, when we evaluate the model with the training data, we see only a slight improvement in the accuracy
        - 87.3% to 88.6%
    - this is a clear case of 'overfitting'
'''
# with hardcoded epoch amount
# model.fit(training_imgs, training_labels, epochs=5)

# with 'callback' function
model.fit(training_imgs, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_imgs, test_labels)

classifications = model.predict(test_imgs)
print(classifications[0])
print(test_labels[0])