import numpy as np

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers):
    self.L = len(layers)    # number of layers
    self.n = layers.copy()  # number of neurons in each layer

    self.xi = []            # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []             # edge weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
