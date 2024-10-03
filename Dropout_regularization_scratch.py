import numpy as np

class Dropout:
    def __init__(self,dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask =None 

    def forward(self,x,is_training=True):
        if is_training:
            #generate a mask that randomly drops out units
            self.mask = np.random.binomial(1,1-self.dropout_rate, size = x.shape) / (1-self.dropout_rate)
            return x * self.mask
        else:
            return x

    #backward pass for dropout
    def backward(self,d_out):
        return d_out * self.mask 
      
#example input data
x = np.array([[1,2,3],[4,5,6]])
dropout_layer = Dropout(dropout_rate=0.5)
output_train = dropout_layer.forward(x, is_training=True)
print(f"Output after Dropout(Training): \n {output_train}")

d_out = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
#example of gradients flowing back
gradient_back = dropout_layer.backward(d_out)
print(f"Gradients: \n {gradient_back}")

#forwaard pass- inference mode
output_infer = dropout_layer.forward(x, is_training=False)
print(f"Output after dropout (inference): \n {output_infer}")
