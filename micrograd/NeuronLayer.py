import math
import random
from .Neuron import Neuron


# Definition class for a fully-connected layer of a multi-layer perceptron
class NeuronLayer:
    def __init__(self, inputCount, neuronCount, activationFunction="sigmoid"):
        # Generate all of the neurons in the layer
        self.neurons = [Neuron(inputCount, activationFunction)
                        for _ in range(neuronCount)]

    # Pass inputs through this layer
    def __call__(self, inputs):
        # Compute the activations for each neuron
        return [neuron(inputs) for neuron in self.neurons]

    # Get all of the parameters for this layer
    def getParameters(self):
        return [parameter
                for neuron in self.neurons
                for parameter in neuron.getParameters()]
