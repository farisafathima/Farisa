# # 1. Implementation of Activation Functions and their Derivatioves from scratch  and plotting the sameusing python
# # a. Sigmoid
# # b. Tanh
# # c. ReLU (Rectified Linear Unit)
# # d. Leaky ReLU
# # e. Softmax

# # 2.The following are the observations from the plot
# # a.  min and max values for the functions
# # b. Are the output of the function zero-centred?
# # c. What happens to the gradient when the input values are too small or too big?
#

import numpy as np
import matplotlib.pyplot as plt


# Functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_derivative(z):
    tanh_z = tanh(z)
    return 1 - tanh_z ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z >= 0, 1, 0)


def leaky_relu(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0)


def softmax_derivative(z):
    s = softmax(z).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# Generate 100 equally spaced values between -10 and 10
z = np.linspace(-10, 10, 100)

# Plot functions and their derivatives
functions = {
    'Sigmoid': (sigmoid, sigmoid_derivative),
    'Tanh': (tanh, tanh_derivative),
    'ReLU': (relu, relu_derivative),
    'Leaky ReLU': (leaky_relu, leaky_relu_derivative),
    'Softmax': (softmax, softmax_derivative)
}

for name, (func, deriv) in functions.items():
    y = func(z)
    dy = deriv(z)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1) 
    plt.plot(z, y, label=f'{name}')
    plt.axvline(x=0, linestyle='dashed', color='grey')
    plt.xlabel('z')
    plt.ylabel(name)
    plt.title(f'Plot of {name}')
    plt.legend()

    # Plot the derivative
    plt.subplot(1, 2, 2)
    if name == 'Softmax':
        # Take a random row from the derivative matrix
        random_row = np.random.choice(dy.shape[0])
        plt.plot(z, dy[random_row], label=f'{name} Derivative (Row {random_row})')
    else:
        plt.plot(z, dy, label=f'{name} Derivative')

    plt.axvline(x=0, linestyle='dashed', color='grey')
    plt.xlabel('z')
    plt.ylabel(f'{name} Derivative')
    plt.title(f'Plot of {name} Derivative')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Observations
def observations(func_name, y):
    print(f"\nObservations for {func_name}:")
    print(f"a. Min and Max values: {y.min()}, {y.max()}")
    print(f"b. Zero-centered: {'Yes' if np.isclose(y.mean(), 0, atol=0.1) else 'No'}")

    gradients = np.gradient(y, z)
    small_values = gradients[z < -5].mean()
    big_values = gradients[z > 5].mean()
    print(f"c. Gradient behavior for small/large inputs: Small: {small_values}, Large: {big_values}")


for name, (func, deriv) in functions.items():
    y = func(z)
    observations(name, y)
