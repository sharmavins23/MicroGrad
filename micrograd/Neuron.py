import math
import random
from .DerivableValue import DerivableValue as DV


# Definition class for a singular neuron of a network
class Neuron:
    def __init__(self, inputCount, activationFunction="sigmoid"):
        # Generate all of the weights
        self.weights = [DV(random.uniform(-1, 1)) for _ in range(inputCount)]

        # Generate the initial bias
        self.bias = DV(random.uniform(-1, 1))

        # Save the activation function
        self.activationFunction = activationFunction

    # Pass an input through this neuron
    def __call__(self, inputs):
        # Compute the activations - Sum of w * x for all weights and inputs
        # Bias is added to the sum in order to shift the activation function
        activation = sum(
            (w * x for w, x in zip(self.weights, inputs)), self.bias)

        # Finally, apply nonlinearity
        return self._applyNonlinearity(activation)

    def _applyNonlinearity(self, activation):
        # Switch on the activation function specified
        if self.activationFunction == "identity":
            return activation.identity()
        elif self.activationFunction == "heaviside":
            return activation.heaviside()
        elif self.activationFunction == "sigmoid":
            return activation.sigmoid()
        elif self.activationFunction == "tanh":
            return activation.tanh()
        elif self.activationFunction == "relu":
            return activation.relu()
        elif self.activationFunction == "gelu":
            return activation.gelu()
        elif self.activationFunction == "softplus":
            return activation.softplus()
        elif self.activationFunction == "selu":
            return activation.selu()
        elif self.activationFunction == "leakyRelu":
            return activation.leakyRelu()
        elif self.activationFunction == "silu":
            return activation.silu()
        elif self.activationFunction == "gaussian":
            return activation.gaussian()

        # Always default to sigmoid
        else:
            return activation.sigmoid()

    def getParameters(self):
        # Return all of the weights and the bias
        return self.weights + [self.bias]
