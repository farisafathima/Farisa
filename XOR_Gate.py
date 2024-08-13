# XOR implementation from Scratch - No deep learning libraries

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Derivative of the sigmoid function
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# Initialize parameters for a 3-layer neural network
def initialize_parameters(layer_sizes):
    parameters = {}
    for l in range(1, len(layer_sizes)):
        parameters['w' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))
    return parameters


# Forward pass through the network
def forward_pass(x, parameters):
    dict = {}
    a = x
    L = len(parameters) // 2  # Number of layers
    for i in range(1, L + 1):
        z = np.dot(parameters['w' + str(i)], a) + parameters['b' + str(i)]
        a = sigmoid(z)
        dict['a' + str(i)] = a
        dict['z' + str(i)] = z
    return dict


# Backward pass to compute gradients
def backprop(x, y_true, dict, parameters):
    grads = {}
    L = len(parameters) // 2  # Number of layers
    m = y_true.shape[1]  # Number of examples

    # Compute gradients for the output layer
    y_hat = dict['a' + str(L)]  # Predictions from the forward pass
    dz_dl = y_hat - y_true  # Error term
    grads['dW' + str(L)] = np.dot(dz_dl, dict['a' + str(L - 1)].T) / m
    grads['db' + str(L)] = np.sum(dz_dl, axis=1, keepdims=True) / m

    # Backpropagate through hidden layers
    for l in reversed(range(1, L)):
        da = np.dot(parameters['w' + str(l + 1)].T, dz_dl)
        dz = da * sigmoid_derivative(dict['z' + str(l)])
        grads['dW' + str(l)] = np.dot(dz, x.T if l == 1 else dict['a' + str(l - 1)].T) / m
        grads['db' + str(l)] = np.sum(dz, axis=1, keepdims=True) / m
        dz_dl = dz  # Update error term for next layer

    return grads


# Update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers
    for l in range(1, L + 1):
        parameters['w' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters


# Compute the loss using binary cross-entropy
def compute_loss(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
    return loss


# Training loop
def train_neural_network(X, y, layer_sizes, learning_rate=0.1, num_iterations=1000):
    parameters = initialize_parameters(layer_sizes)
    losses = []

    for i in range(num_iterations):
        # Forward pass
        dict = forward_pass(X, parameters)

        # Compute predictions
        y_pred = dict['a' + str(len(layer_sizes) - 1)]

        # Compute loss
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        # Backward pass
        grads = backprop(X, y, dict, parameters)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Loss {loss}")

    return parameters, losses


# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Input features
y = np.array([[0], [1], [1], [0]]).T  # Output labels

# Define layer sizes (2 inputs, 4 hidden neurons, 1 output)
layer_sizes = [2, 4, 1]

# Train the neural network
parameters, losses = train_neural_network(X, y, layer_sizes, learning_rate=0.1, num_iterations=1000)

# Plot training loss
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.show()

print("Updated parameters:")
print(parameters)
