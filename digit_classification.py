from network import Network
import numpy as np

if __name__ == '__main__':
    # load and preprocess data
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    x_test = np.load('data/x_test.npy')
    y_test = np.load('data/y_test.npy')

    # one hot encode y
    b = np.zeros((y_train.size, y_train.max() + 1))
    b[np.arange(y_train.size), y_train] = 1
    y_train = b

    b = np.zeros((y_test.size, y_test.max() + 1))
    b[np.arange(y_test.size), y_test] = 1
    y_test = b

    # unravel x data
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)

    # zip x and y data
    train_xy = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_train, y_train)]
    test_xy = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(x_test, y_test)]

    n = Network([28*28, 30, 10])
    n.train(train_xy, epochs=30, mini_batch_size=100, eta=1.0, test_data=test_xy)
    # save weights and biases to file
    weights = n.weights
    biases = n.biases
    for layer in range(len(weights)):
        np.save(f'weights/weights{layer}', weights[layer])
        np.save(f'weights/biases{layer}', biases[layer])