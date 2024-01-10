import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import warnings

warnings.simplefilter('ignore')

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class Network:
  def __init__(self, sizes, activation_function=sigmoid):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.activation_function = activation_function
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    self.activations = [np.zeros((s, 1)) for s in self.sizes]
    self.zs = [np.zeros((s, 1)) for s in self.sizes[1:]]

  def feedforward(self, a):
    a = np.reshape(a, (len(a), 1)) # make sure input, as long as list-like is formatted as a column vector
    for i, (b, w) in enumerate(zip(self.biases, self.weights)):
      z = np.dot(w, a) + b
      a = self.activation_function(z)
      self.activations[i+1] = a
      self.zs[i] = z
    return a

  def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [
          training_data[k:k+mini_batch_size]
          for k in range(0, n, mini_batch_size)
      ]

      for batch in mini_batches:
        self.update_mini_batch(batch, eta)

      if test_data:
        accuracy = self.evaluate(test_data) / n_test
        print(f'Accuracy following Epoch {j+1}: {accuracy}')

  def update_mini_batch(self, mini_batch, eta):
    #nabla = ∇
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+db for nb, db in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dw for nw, dw in zip(nabla_w, delta_nabla_w)]

    self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    return

  def backprop(self, x, y):

    x, y = np.reshape(x, (len(x), -1)), np.reshape(y, (len(y), -1))

    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    activation = x
    self.activations[0] = x
    #forward pass
    self.feedforward(activation)

    #backward pass
    delta = (self.activations[-1] - y) * sigmoid_prime(self.zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, self.activations[-2].T)

    for l in range(2, self.num_layers):
      z = self.zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].T, delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, self.activations[-l-1].T)

    return nabla_b, nabla_w
  
  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  def __repr__(self):
    return f'<Network {self.sizes}>'

