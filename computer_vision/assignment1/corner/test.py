import numpy as np
import matplotlib.pyplot as plt
import cv2

im=cv2.imread('./chessboard.jpg',0)
im=cv2.resize(im,None,fx=0.5,fy=0.5)

cv2.imwrite('./chessboard.jpg',im)