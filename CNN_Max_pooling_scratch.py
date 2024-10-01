#This script contains CNN implementation from scratch


import numpy as np

def apply_padding(d2_image, padding_num):
    if padding_num > 0:
        return np.pad(d2_image, ((padding_num,padding_num),(padding_num,padding_num)), mode='constant')
    return d2_image

def convolution_operation(input_image,kernel,stride=1,padding=0):
    input_image_padded = apply_padding(input_image,padding)

    output_dim =((input_image_padded.shape[0] - kernel.shape[0])// stride) +1

    output_image = np.zeros((output_dim,output_dim))

    for i in range(0,output_dim):
        for j in range(0,output_dim):
            region = input_image_padded[i * stride: i * stride + kernel.shape[0], j * stride:j * stride + kernel.shape[1]]
            conv_value = np.sum(region * kernel)
            output_image[i,j] = conv_value
    return output_image


def max_pooling(input_image,pool_size=2,stride = 2):
    output_dim =((input_image.shape[0] - pool_size )//stride ) +1
    pooled_image = np.zeros((output_dim,output_dim))
    for i in range(0, output_dim):
        for j in range(0, output_dim):
            region = input_image[i*stride : i*stride+pool_size, j*stride : j* stride + pool_size]
            pool_val = np.max(region)
            pooled_image[i,j] = pool_val

    return pooled_image


input_image = np.random.rand(5,5)
kernel = np.random.rand(3,3)
stride=1
padding =1
pool_size =2
pool_stride =2

conv_output = convolution_operation(input_image,kernel,padding=padding,stride=stride)
print("Convolution output\n", conv_output)

max_output = max_pooling(input_image, pool_size=pool_size, stride=pool_stride)
print("\nMax pooling output\n",max_output)

