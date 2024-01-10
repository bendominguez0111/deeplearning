from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.save('data/x_train', x_train)
np.save('data/y_train', y_train)
np.save('data/x_test', x_test)
np.save('data/y_test', y_test)