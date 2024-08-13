#This is the implementation of Backpropagation from scratch without using any libraries


import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def sig_der(z):
    s = sigmoid(z)
    return s * (1 - s)

# Initialize parameters for a multi-layer neural network
def initialize_param(layer_sizes):
    param = {}
    for l in range(1, len(layer_sizes)):
        # Initialize weights with small random values and biases with zeros
        param['w' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.01
        param['b' + str(l)] = np.zeros((layer_sizes[l], 1))
    return param

# Forward pass through the network
def forward_pass(x, param):
    dict = {}
    a = x
    L = len(param) // 2  # Number of layers
    for i in range(1, L + 1):
        # Compute linear combination of inputs and weights, add bias
        z = np.dot(param['w' + str(i)], a) + param['b' + str(i)]
        # Apply sigmoid activation function
        a = sigmoid(z)
        # Store activations and linear combinations in dict
        dict['a' + str(i)] = a
        dict['z' + str(i)] = z
    return dict

# Backward pass to compute gradients
def backprop(x, y_true, dict, param):
    grads = {}
    L = len(param) // 2  # Number of layers
    m = y_true.shape[1]

    # Compute gradients for the output layer
    y_hat = dict['a' + str(L)]  # Predictions from the forward pass
    dz_dl = y_hat - y_true  # Error term
    grads['dW' + str(L)] = np.dot(dz_dl, dict['a' + str(L - 1)].T) / m
    grads['db' + str(L)] = np.sum(dz_dl, axis=1, keepdims=True) / m

    # Print gradients for the output layer
    print(f"Layer {L} gradients:")
    print(f"dW{L}: {grads['dW' + str(L)]}")
    print(f"db{L}: {grads['db' + str(L)]}")

    # Backpropagate through hidden layers
    for l in reversed(range(1, L)):
        # Compute gradient of the activation for the current layer
        da = np.dot(param['w' + str(l + 1)].T, dz_dl)
        dz = da * sig_der(dict['z' + str(l)])
        # Compute gradients for weights and biases
        grads['dW' + str(l)] = np.dot(dz, x.T if l == 1 else dict['a' + str(l - 1)].T) / m
        grads['db' + str(l)] = np.sum(dz, axis=1, keepdims=True) / m
        dz_dl = dz  # Update error term for next layer

        # Print gradients for each hidden layer
        print(f"Layer {l} gradients:")
        print(f"dW{l}: {grads['dW' + str(l)]}")
        print(f"db{l}: {grads['db' + str(l)]}")

    return grads

# Update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers
    for l in range(1, L + 1):
        parameters['w' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters

# Example input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
y = np.array([[0], [1], [1], [0]]).T

# Define layer sizes (2 inputs, 2 hidden layers with 4 and 3 neurons, 1 output)
layer_sizes = [2, 4, 3, 1]

# Initialize parameters
parameters = initialize_param(layer_sizes)

# Perform a forward pass
ffn = forward_pass(X, parameters)

# Perform backpropagation and print gradients
grads = backprop(X, y, ffn, parameters)

# Update parameters
parameters = update_parameters(parameters, grads, learning_rate=0.1)

print("Updated parameters:")
print(parameters)
