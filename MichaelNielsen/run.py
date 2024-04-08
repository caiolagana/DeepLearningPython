# Based on http://neuralnetworksanddeeplearning.com/chap1.html

from mnist_loader import *
from network import *

net = Network([784, 30, 10])
training_data, validation_data, test_data = load_data_wrapper()
net.SGD(
    training_data=training_data,
    epochs=3,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)