# Batch Normalization from Scratch
import numpy as np

#technique that normalizes input of each layer so that they have zero mean and unit variance--speed up training, better convergence

def batch_norm(x, gamma, beta, eps=1e-5):
    #gamma -scale parameter
    #beta - shift parameter

    mean = np.mean(x,axis=0)
    variance = np.var(x,axis=0)
    #variance measures how much values differ from mean

    x_normalized = (x-mean)/ np.sqrt(variance+eps)

    #scale and shift
    normalised_x = gamma * x_normalized + beta
    return normalised_x

x = np.random.randn(4,3)
beta = np.zeros(3)
gamma = np.ones(3)
output = batch_norm(x,gamma,beta)
print("Batch normalization output\n", output)

# Layer Normalization from Scratch


import numpy as np


#technique that normalizes across the features of each individual sample, rather than across a batch
#commonly used for rnn
def layer_normalization(x,gamma,beta , eps=1e-5):
    mean = np.mean(x,axis=1, keepdims=True)
    #keepdims - ensures that the dimensions are retained
    variance = np.var(x, axis=1, keepdims=True)
    x_normalized = (x-mean)/ np.sqrt(variance+eps)
    normalized_x = gamma * x_normalized + beta
    return normalized_x

x = np.random.randn(4,3)
gamma = np.ones((1,3))
beta = np.zeros((1,3))
output = layer_normalization(x,gamma,beta)
print("Layer normalization output\n", output)

