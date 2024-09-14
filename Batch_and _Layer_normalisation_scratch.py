# Batch Normalization from Scratch
import numpy as np


def batch_norm(X, gamma, beta, eps=1e-5):

    # Calculate the mean and variance across the batch
    mean = np.mean(X, axis=0)  
    variance = np.var(X, axis=0)  

    #  Normalize the batch
    X_normalized = (X - mean) / np.sqrt(variance + eps)

    # Scale and shift using gamma and beta
    normalized_X = gamma * X_normalized + beta

    return normalized_X



X = np.random.randn(4, 3)  
# print(X)
gamma = np.ones(3)  
beta = np.zeros(3)  
output = batch_norm(X, gamma, beta)
print("Batch Normalization Output:")
print(output)

# Layer Normalization from Scratch


def layer_norm(X, gamma, beta, eps=1e-5):

    # Calculate mean and variance for each individual input
    mean = np.mean(X, axis=1, keepdims=True)  # Mean for each sample
    variance = np.var(X, axis=1, keepdims=True)  # Variance for each sample

    #Normalize each input
    X_normalized = (X - mean) / np.sqrt(variance + eps)

    # Scale and shift using gamma and beta
    normalized_X = gamma * X_normalized + beta

    return normalized_X



X = np.random.randn(4, 3)  
gamma = np.ones((1, 3))  
beta = np.zeros((1, 3)) 
output = layer_norm(X, gamma, beta)
print("\nLayer Normalization Output:")
print(output)

