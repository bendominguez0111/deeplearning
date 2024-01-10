import numpy as np

def function(x):
    return x**2

def derivative_function(x):
    return 2*x

x = np.arange(-100, 100, 0.1)
y = function(x)

x_i = 10
current_position = x_i, function(x_i)

learning_rate = 0.01
num_iterations = 1_000

for i in range(num_iterations):
    g = derivative_function(current_position[0])
    new_x = current_position[0] - learning_rate*g
    new_y = function(new_x)

    current_position = (new_x, new_y)

    if i % 100 == 0:
        distance_from_optimal = np.sqrt(new_x**2 + new_y**2)
        print('Current Position: ', current_position)
        print('Distance from optimal position: ', distance_from_optimal)