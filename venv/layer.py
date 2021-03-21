from functions import sigmoid
import numpy as np


class Layer:

    def __init__(self, num_inputs, num_outputs, activation=sigmoid):
        self.weights = np.random.uniform(-np.sqrt(num_inputs), np.sqrt(num_inputs),
                                         size=(num_inputs, num_outputs))
        self.biases = np.random.uniform(-np.sqrt(num_inputs), np.sqrt(num_inputs),
                                        size=num_outputs)

    def getWeights(self):
        return self.weights

    def get