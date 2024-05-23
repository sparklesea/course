import numpy as np

zero=np.zeros((2,3,2))
print(zero)
zero[0][2][0]=1
zero[1][2][0]=1
print(zero)
zero=np.reshape(zero,(-1,6))
print(zero)