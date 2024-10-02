import torch

#optimizers are algorithms used to minimize the loss function by adjusting model parameters

print("\nOptimization process using PyTorch")

#create an initial parameter
initial_param = torch.randn(3, requires_grad=True)

#cretae a clone for each optimization technique
param_sgd = initial_param.clone().detach().requires_grad_(True)
param_momentum = initial_param.clone().detach().requires_grad_(True)
param_adagrad = initial_param.clone().detach().requires_grad_(True)

#create optimizers
sgd_optimizer = torch.optim.SGD([param_sgd],lr = 0.01)
momentum_optimizer = torch.optim.SGD([param_momentum], lr =0.01, momentum=0.9)
adagrad_optimizer = torch.optim.Adagrad([param_adagrad], lr = 0.01)

#loss function
loss_fn = torch.nn.MSELoss()


#example data target
input_data = torch.randn(3)
target = torch.randn(3)

output_sgd = param_sgd * input_data
loss_sgd = loss_fn(output_sgd,target)
loss_sgd.backward()
sgd_optimizer.step()
sgd_optimizer.zero_grad()


output_momentum =param_momentum * input_data
loss_momentum = loss_fn(output_momentum, target)
loss_momentum.backward()
momentum_optimizer.step()
momentum_optimizer.zero_grad()


output_adagrad = param_adagrad * input_data
loss_adagrad = loss_fn(output_adagrad, target)
loss_adagrad.backward()
adagrad_optimizer.step()
adagrad_optimizer.zero_grad()

#print updated parameters
print("SGD parameters:", param_sgd)
print("Momentum parameters:", param_momentum)
print("Adagrad parameters:", param_adagrad)


