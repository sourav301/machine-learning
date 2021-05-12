#Training a multi layered perceptron using nn module of pytorch

import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

lr=.2
inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float)
outputs = torch.tensor([[0],[1],[1],[0]],dtype=torch.float)

m = inputs.size(0)


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron,self).__init__()
        self.lin1 = nn.Linear(2, 2,bias=True) 
        self.lin2 = nn.Linear(2, 1,bias=True) 

    def forward(self, x): 
        x = torch.sigmoid(self.lin1(x))
        x =  self.lin2(x) 
        return x
    
    
loss = torch.nn.MSELoss()
model = MultiLayerPerceptron()
optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.8)
losses=[]
for i in range(1000):
    
    pred = model(inputs)
    l=loss(pred,outputs)
    l.backward()
    optimizer.step()
    
    optimizer.zero_grad()
    losses.append(l.item())
plt.plot(losses)
print(pred)
print(pred.round())  
print(losses[-1])