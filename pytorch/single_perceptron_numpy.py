# Traninig a single layered perceptron using numpy

import numpy as np

lr=.2
X=[[0,0],[0,1],[1,0],[1,1]]
X = np.array(X)
X = np.hstack((np.ones((4,1)),X))

Y = [[0],[1],[1],[1]]
Y = np.array(Y)

m = X.shape[0]
n = X.shape[1]

W = np.random.random((3,1)) 


for i in range(100):
    H = X @ W
    D = H - Y
    delta = 1/m*(X.T @ D)
    
    W = W - lr*delta 
    
print(X @ W,"\n")