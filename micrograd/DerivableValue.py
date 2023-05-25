import math
import random


# Definition class for a derivable value
class DerivableValue:
    def __init__(self, data, children=()):
        # We only accept numbers as data
        assert isinstance(data, (int, float)), f"Data {data} must be a number!"
        self.data = data

        # Default any and all gradients to 0.0
        self.grad = 0.0

        # Set a default derivation function to None
        self._updatePartialDerivative = lambda: None

        # Keep track of any children
        self._children = children

    # ===== Core mathematical functions (with derivatives) =====================

    # Add two values
    def __add__(self, other):
        # Sanity check with the value types
        other = other if isinstance(
            other, DerivableValue) else DerivableValue(other)

        # Create a new value with the sum of the data
        out = DerivableValue(self.data + other.data, (self, other))

        def _updatePartialDerivative():
            # Partial derivative of addition is 1 in both cases
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Multiply two values
    def __mul__(self, other):
        # Sanity check with the value types
        other = other if isinstance(
            other, DerivableValue) else DerivableValue(other)

        # Create a new value with the product of the data
        out = DerivableValue(self.data * other.data, (self, other))

        def _updatePartialDerivative():
            # Partial derivative of multiplication is the other value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Compute the exponential e^x of a value
    def exp(self):
        # Create a new value with the exponential of the data
        out = DerivableValue(math.exp(self.data), (self,))

        def _updatePartialDerivative():
            # Partial derivative of exponential is the exponential
            self.grad += out.data * out.grad  # No need to recompute
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Compute the power of two values
    def __pow__(self, other):
        # Sanity check with the value types
        other = other if isinstance(
            other, DerivableValue) else DerivableValue(other)

        # Create a new value with the power of the data
        out = DerivableValue(self.data**other.data, (self, other))

        def _updatePartialDerivative():
            # Partial derivative for x^n = n * x^(n-1)
            self.grad += other.data * \
                (self.data ** (other.data - 1.0)) * out.grad
            # Partial derivative for a^x = a^x * ln(a)
            if (self.data > 0.0):
                other.grad += (self.data ** other.data) * \
                    math.log(self.data) * out.grad
            else:
                other.grad += 0.0 * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # ===== Neuron activation functions ========================================

    # Identity activation
    def identity(self):
        out = DerivableValue(self.data, (self,))

        def _updatePartialDerivative():
            self.grad += 1.0 * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Binary step activation
    def heaviside(self):
        out = DerivableValue(0.0 if self.data < 0.0 else 1.0, (self,))

        def _updatePartialDerivative():
            self.grad += 0.0 * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Logistic activation
    def sigmoid(self):
        out = DerivableValue(1.0 / (1.0 + math.exp(-self.data)), (self,))

        def _updatePartialDerivative():
            self.grad += (out.data * (1.0 - out.data)) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Hyperbolic tangent activation
    def tanh(self):
        out = DerivableValue(math.tanh(self.data), (self,))

        def _updatePartialDerivative():
            self.grad += (1.0 - (out.data ** 2.0)) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Rectified linear unit activation
    def relu(self):
        out = DerivableValue(max(0.0, self.data), (self,))

        def _updatePartialDerivative():
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Gaussian error linear unit activation
    def gelu(self):
        out = DerivableValue(0.5 * self.data *
                             (1.0 + math.erf(self.data / math.sqrt(2.0))), (self,))

        def _updatePartialDerivative():
            self.grad += (0.5 * (1.0 + math.erf(self.data / math.sqrt(2.0))) +
                          (math.exp(-0.5 * (self.data ** 2.0)) / math.sqrt(2.0 * math.pi))) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Softplus activation
    def softplus(self):
        out = DerivableValue(math.log(1.0 + math.exp(self.data)), (self,))

        def _updatePartialDerivative():
            self.grad += (1.0 / (1.0 + math.exp(-self.data))) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Scaled exponential linear unit activation
    def selu(self):
        out = DerivableValue(1.0507 * self.data *
                             (1.0 if self.data > 0.0 else 1.67326 * math.exp(self.data) - 1.67326), (self,))

        def _updatePartialDerivative():
            self.grad += (1.0507 * (1.0 if self.data > 0.0 else 1.67326 *
                                    math.exp(self.data))) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Leaky rectified linear unit activation
    def leakyRelu(self, alpha=0.01):
        out = DerivableValue(max(alpha * self.data, self.data), (self,))

        def _updatePartialDerivative():
            self.grad += (1.0 if self.data > 0.0 else alpha) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Sigmoid linear unit activation (Sigmoid Shrinkage)
    def silu(self):
        out = DerivableValue(self.data / (1.0 + math.exp(-self.data)), (self,))

        def _updatePartialDerivative():
            self.grad += (out.data + (math.exp(-self.data) /
                                      ((1.0 + math.exp(-self.data)) ** 2.0))) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # Gaussian activation
    def gaussian(self):
        out = DerivableValue(math.exp(-0.5 * (self.data ** 2.0)), (self,))

        def _updatePartialDerivative():
            self.grad += (-self.data * math.exp(-0.5 *
                          (self.data ** 2.0))) * out.grad
        out._updatePartialDerivative = _updatePartialDerivative

        return out

    # ===== Helpers and function compositions ==================================

    # Add two values (inverted)
    def __radd__(self, other):
        return self + other

    # Subtract two values
    def __sub__(self, other):
        # Subtraction is simply addition with a negated value
        return self + (-other)

    # Subtract two values (inverted)
    def __rsub__(self, other):
        # Subtraction is simply addition with a negated value
        return other + (-self)

    # Multiply two values (inverted)
    def __rmul__(self, other):
        return self * other

    # Negate a value
    def __neg__(self):
        # Negation is simply multiplication with -1
        return self * -1

    # Divide two values
    def __truediv__(self, other):
        # Division is simply multiplication with the reciprocal power
        return self * (other**-1)

    # Divide two values (inverted)
    def __rtruediv__(self, other):
        # Division is simply multiplication with the reciprocal power
        return other * (self**-1)

    # Write out a string representation of the value
    def __repr__(self):
        return f"DV({self.data} | {self.grad})"

    # ===== Derivation =========================================================

    # Compute the backwards propagated chain rule derivation for a value
    def computeFullDerivative(self):
        topologicalSort = []
        visited = set()

        # Build a topological sort of the graph recursively
        def buildTopologicalSort(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    buildTopologicalSort(child)
                topologicalSort.append(node)

        # Build the topological sort starting with the current node
        buildTopologicalSort(self)

        # Now compute the partial derivatives for each node in the sort
        self.grad = 1.0
        for node in reversed(topologicalSort):
            node._updatePartialDerivative()
