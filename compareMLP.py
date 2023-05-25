# Compare and contrast MLP activation functions!
import matplotlib.pyplot as plt
import random
import time

from micrograd.MultilayerPerceptron import MultilayerPerceptron as MLP

# Set a random seed for reproducibility
random.seed(23)


# ===== Parameters and tuning ==================================================

# Number of neurons in each hidden layer (and output)
hiddenLayerStructure = [10, 5, 4, 1]

# Training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# Target data
ys = [1.0, -1.0, -1.0, 1.0]

# Create a list of activation functions to test
activationFunctions = [
    "identity",
    "heaviside",
    "sigmoid",
    "tanh",
    "relu",
    "gelu",
    "softplus",
    "selu",
    "leakyRelu",
    "silu",
    "gaussian"
]

# Learning rate
learningRate = 0.001

# Epoch count (number of times we repeat the training data)
epochCount = 1000


# ===== Driver code ============================================================


# Function to test a particular activation function
def testActivationFunction(activationFunction):
    # Create a network with the given activation function
    n = MLP(3, hiddenLayerStructure, activationFunction)

    # We'll simply train the network over this data n times
    losses = []
    for _ in range(epochCount):
        loss = n.train(xs, ys, learningRate)
        losses.append(loss)

    # Return the losses
    return losses


# Driver code
if __name__ == "__main__":
    # Iterate through all of the activation functions
    allLosses = []
    for activationFunction in activationFunctions:
        losses = testActivationFunction(activationFunction)
        allLosses.append(losses)

    # Plot the losses, all on the same graph, labelled
    for i in range(len(activationFunctions)):
        plt.plot(allLosses[i], label=activationFunctions[i])

    # Add a title and labels
    plt.title("Losses for different activation functions")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    # Set yMax to 10.0
    plt.ylim(-1.0, 10.0)

    # Add a legend and show the graph
    plt.legend()
    plt.show()
