from functions import sigmoid
import numpy as np


class Neuron:

    def __init__(self, random_border=1, activation=None):
        random_border = np.sqrt(random_border)
        self.weight = np.random.randint(random_border, random_border)
        self.bias = np.random.randint(random_border, random_border)
        self.activation = functions.sigmoid()