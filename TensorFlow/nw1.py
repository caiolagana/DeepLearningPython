# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Following tutorial: https://www.tensorflow.org/tutorials/keras/classification

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
print(train_images.shape)
(60000, 28, 28)
60k matrices of 28x28 --> 60k images of 28x28 pixels

print(len(train_labels))
--> 10
"""

"""Normalize each pixel's value to lie between [0,1], instead of original [0,255]"""
train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=2)


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
print(np.argmax(predictions_single[0]))