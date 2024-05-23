import numpy as np
import matplotlib.pyplot as plt
import pyrtools as pt

img_tmp=plt.imread('./Einstein.jpg')
img_curie=plt.imread('./Curie.pgm')
r,g,b=[img_tmp[:,:,i] for i in range(3)]
img_einstein=(r*0.299+g*0.587+b*0.114).astype(np.uint8)#该图片为3通道，需要进行预处理

def blur(input,level,filter='binom5'):#blur函数，可以指定模糊的guassian pyramid level
    input=pt.blurDn(input,level,filter)
    return pt.upBlur(input,level,filter)

blurred1=0.75*blur(img_curie,1)
blurred2=blur(img_curie,2)
blurred3=blur(img_curie,3)
fine1=img_einstein-2*blur(img_einstein,1)
fine2=img_einstein-4*blur(img_einstein,2)
fine3=img_einstein-blur(img_einstein,3)
hybrid1=blurred1+fine1
hybrid2=blurred2+fine2
hybrid3=blurred3+fine3
pt.imshow([blurred1,blurred2,blurred3])
pt.imshow([fine1,fine2,fine3])
pt.imshow([hybrid1,hybrid2,hybrid3])

plt.imsave('hybrid1.png',hybrid1,cmap='gray')
plt.imsave('hybrid2.png',hybrid2,cmap='gray')
plt.imsave('hybrid3.png',hybrid3,cmap='gray')