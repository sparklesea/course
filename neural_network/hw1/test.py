import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
x=torch.tensor([[0,1,0,5]],dtype=torch.float32)
y=torch.tensor([3])
print(x.dtype)
print(y.dtype)
print(criterion(x,y))

x=np.linspace(-8,8,300)
y=np.linspace(-8,8,300)
print(x,y)
X,Y=np.meshgrid(x,y)
print(X,Y)

plt.plot(X,Y,marker='.',color='blue')
plt.show()