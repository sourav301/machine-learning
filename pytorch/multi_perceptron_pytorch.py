# Implement 2 layer perceptron using torch tensor and using autograd
# XOR gate has been trained showing the ability of the model to make a non-linear decision boundary.

import torch 
import matplotlib.pyplot as plt

lr=.2 
inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float)
outputs = torch.tensor([[0],[1],[1],[0]],dtype=torch.float)

# inputs = torch.tensor([[0.,0.],[1.,1.]])
# outputs = torch.tensor([[0.],[1.]])

m = inputs.size(0)
weights1 = torch.randn(2,2, requires_grad=True)
weights2 = torch.randn(2,1, requires_grad=True)
bias1 = torch.randn(2, requires_grad=True)
bias2 = torch.randn(1, requires_grad=True)
losses =[]
for _ in range(3000):
     
    sum1 = inputs @ weights1 + bias1
    
    z1 = sum1.sigmoid()
     
    sum2 = z1 @ weights2 + bias2
    z2 = sum2.sigmoid()
    
    loss = (z2 - outputs)**2 
    loss.sum().backward(retain_graph=True)
    
    with torch.no_grad(): 
        bias1-=(lr*bias1.grad)
        weights1-=(lr*weights1.grad)
        
        bias2-=(lr*bias2.grad)
        weights2-=(lr*weights2.grad)
    
    bias1.grad.zero_()
    bias2.grad.zero_()
    weights1.grad.zero_()
    weights2.grad.zero_() 
    losses.append(loss.sum().item())
print(z2.round()) 
plt.plot(losses)
print(losses[-1])


# for _ in range(10):
#     pred = inputs @ weights
#     pred = torch.unsqueeze(pred,1)
    
#     er
#     error = outputs-pred
    
    
#     weights.add_(lr*sum(sum(error)*inputs)/m)
    
#     print(abs(round(sum(error).item(),5) ))
    

# print(pred)