# W is a matrix, X is a vector, z is a vector, and a is a vector. y^ is a scalar and a final prediction.
# a is ReLU(z). Initialize X and W randomly

import numpy as np

def activation_function(x):
    # return np.maximum(0, x)  # Optimized ReLU using numpy
    return  1/(1+np.exp(-x))
np.random.seed(42)
num_of_layers = 2
input_neuron_num = 2
output_neuron_num = 1

# Hidden layer
hidden_layer_num = 1
hl_neuron_num_list = []
for i in range(hidden_layer_num):
    val = input(f"The hidden_layer_{i} num of neurons are: ")
    hl_neuron_num_list.append(int(val))

# Initialize input vector x
x = np.random.uniform(size=(input_neuron_num, 1))
# x = np.array([[0],[1]])
# Initialize weights and biases for each layer
weights = []
biases = []

input_size = input_neuron_num
for i in range(hidden_layer_num):
    hl_neurons = hl_neuron_num_list[i]
    W = np.random.uniform(size=(hl_neurons, input_size))
    # W= np.array([[20,20],[-20,-20]])
    B = np.random.uniform(size=(hl_neurons, 1))
    # B =np.array([[-10],[30]])
    weights.append(W)
    biases.append(B)
    input_size = hl_neurons

# Initialize weights and biases for the output layer
W_out = np.random.uniform(size=(output_neuron_num, input_size))
# W_out = np.array([20,20])
B_out = np.random.uniform(size=(output_neuron_num, 1))
# B_out = np.array([[-30]])

# Feed-forward pass
a = x
for i in range(hidden_layer_num):
    W = weights[i]
    B = biases[i]
    z = np.dot(W,a) + B
    a = activation_function(z)
    print(f"The activation function value of layer {i} is {a}")


# Output layer computation
z_out = np.dot(W_out, a) + B_out
y_hat = activation_function(z_out).round(2)  

print(f"The final prediction is {y_hat}")


