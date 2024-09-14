#This script contains CNN implementation from scratch

import numpy as np


def apply_padding(image, padding):
    if padding > 0:
        return np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    return image


def convolution_operation(input_image, kernel, stride=1, padding=0):
    # Apply padding 
    input_image_padded = apply_padding(input_image, padding)

    output_dim = ((input_image_padded.shape[0] - kernel.shape[0]) // stride) + 1

    output_image = np.zeros((output_dim, output_dim))

    # Perform the convolution operation
    for i in range(0, output_dim):
        for j in range(0, output_dim):
            region = input_image_padded[i * stride:i * stride + kernel.shape[0],
                     j * stride:j * stride + kernel.shape[1]]

            # Perform element-wise multiplication and sum the results
            conv_value = np.sum(region * kernel)

            #convolved value to the output image
            output_image[i, j] = conv_value

    return output_image


def max_pooling_operation(input_image, pool_size=2, stride=2):
    output_dim = ((input_image.shape[0] - pool_size) // stride) + 1

    pooled_image = np.zeros((output_dim, output_dim))

    for i in range(0, output_dim):
        for j in range(0, output_dim):
            region = input_image[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]

            # Find the maximum value in the region
            max_value = np.max(region)

            # maximum value to the pooled image
            pooled_image[i, j] = max_value

    return pooled_image


input_image = np.random.rand(5, 5)  
kernel = np.random.rand(3, 3)  
stride = 1
padding = 1  
pool_size = 2
pool_stride = 2

conv_output = convolution_operation(input_image, kernel, stride=stride, padding=padding)
print("Convolution Output:")
print(conv_output)

pool_output = max_pooling_operation(conv_output, pool_size=pool_size, stride=pool_stride)
print("Max Pooling Output:")
print(pool_output)
