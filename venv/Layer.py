from neuron import Neuron
from functions import sigmoid
import numpy as np


class Layer:

    def __init__(self, num_neurons, num_inputs, num_outputs, activation=sigmoid):
        self.num_neurons = num_neurons
        self.neurons = [Neuron(num_neurons, activation) for _ in range (num_neurons)]
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


