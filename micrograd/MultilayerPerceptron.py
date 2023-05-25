import math
import random
from .NeuronLayer import NeuronLayer


# Definition class for a multilayer perceptron neural network
class MultilayerPerceptron:
    def __init__(self, inputCount, layerSizes, activationFunction="sigmoid"):
        # Create the layers based on stitching the layer sizings together
        self.layerSizeList = [inputCount] + layerSizes
        self.layers = [NeuronLayer(self.layerSizeList[i],
                                   self.layerSizeList[i + 1],
                                   activationFunction)
                       for i in range(len(layerSizes))]

    # Pass an input through the network
    def __call__(self, inputs):
        # Pass the input through each layer
        for layer in self.layers:
            inputs = layer(inputs)

        # Special case: If the output is a single value, return the value itself
        if self.layerSizeList[-1] == 1:
            return inputs[0]

        # At the final layer we should be done
        return inputs

    # Get all of the parameters for this network
    def getParameters(self):
        return [parameter
                for layer in self.layers
                for parameter in layer.getParameters()]

    # Perform one "training pass" of the network
    def train(self, inputs, expectedOutputs, learningRate=0.01):
        # Forward pass
        predictedOutputs = [self(inp) for inp in inputs]
        # Compute the loss of the network - MSE (but no mean)
        loss = sum((predictedOutput - expectedOutput)**2 for predictedOutput,
                   expectedOutput in zip(predictedOutputs, expectedOutputs))

        # Backwards pass
        for parameter in self.getParameters():
            parameter.grad = 0.0
        loss.computeFullDerivative()

        # Apply the gradients (as a descent)
        for parameter in self.getParameters():
            parameter.data -= learningRate * parameter.grad

        return loss.data  # Return the loss for logging purposes
