import numpy as np
import math

# simple linear classifier
# f(x) = w * x + b

# softmax

def softmax_activation(z):
    # z represents the non-normalized weights
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    probabilities = exp_z / sum_exp_z
    return probabilities

def binary_softmax_loss(Y, Y_hat):
    # Y = true labels
    # Y_hat = predicted probabilities after softmax activation
    # m = num of training examples

    m = Y.shape[0]

    component_1 = Y * np.log(Y_hat) # if Y = 0, then this is cancelled
    component_0 = (1 - Y) * np.log(1 - Y_hat) # if Y = 1, then this is cancelled
    loss = component_1 + component_0

    return -np.sum(loss) / m # compute average loss

def gradient_descent(X, Y, num_iterations, learning_rate):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    Y = Y.reshape(m, 1)

    for i in range(num_iterations):
        Z = np.dot(X, w) + b
        Y_hat = softmax_activation(Z)

        dw = np.dot(X.T, (Y_hat - Y)) / m
        db = np.sum(Y_hat - Y) / m

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {binary_softmax_loss(Y, Y_hat)}")
    
    return w, b
    
if __name__ == '__main__':
    X = np.array([ 
        [0.2, 0.3],
        [0.1, 0.3],
        [10.8, 10.11],
        [9.87, 9.05]
    ])

    Y = np.array([0, 0, 1, 1])

    num_iterations = 1000
    learning_rate = 0.01

    w, b = gradient_descent(X, Y, num_iterations, learning_rate)
    print(f"Learned weights: {w}, Bias: {b}")