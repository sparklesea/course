import numpy as np
import matplotlib.pyplot as plt
import cv2

im = cv2.imread('./chessboard.jpg', 0)
im = np.array(im, dtype=np.float32) / 255
im = cv2.resize(im, None, fx=0.5, fy=0.5)


def harris_corner_detector(im):
    filter_x = np.array([[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]], dtype=np.float32)
    filter_y = np.array([[-1, -3, -1], [0, 0, 0], [1, 3, 1]], dtype=np.float32)

    Ix = cv2.filter2D(im, -1, filter_x)
    Iy = cv2.filter2D(im, -1, filter_y)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Sx2 = cv2.GaussianBlur(Ix2, (5, 5), 0.5)
    Sy2 = cv2.GaussianBlur(Iy2, (5, 5), 0.5)
    Sxy = cv2.GaussianBlur(Ixy, (5, 5), 0.5)

    k = 0.5
    im_R = np.zeros_like(im)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            H = np.array([[Sx2[i, j], Sxy[i, j]], 
                          [Sxy[i, j], Sy2[i, j]]])

            im_R[i, j] = np.linalg.det(H) - k * H.trace()

    threshold = np.percentile(im_R, 99)
    corner = np.zeros_like(im_R)
    for i in range(corner.shape[0]):
        for j in range(corner.shape[1]):
            if im_R[i][j] >= threshold:
                if j==corner.shape[1]-1:
                    if im_R[i][j]>im_R[i][j-1]:
                        corner[i][j]=1
                elif j==0:
                    if im_R[i][j]>im_R[i][j+1]:
                        corner[i][j]=1
                elif im_R[i][j]>im_R[i][j+1] and im_R[i][j]>im_R[i][j-1]:
                    corner[i][j] = 1

    return corner + im/5


im_scale = cv2.resize(im, None, fx=0.5, fy=0.5)

M_rotation = cv2.getRotationMatrix2D((im.shape[0] / 2, im.shape[1] / 2), 45, 1)
im_rotation = cv2.warpAffine(im, M_rotation, (im.shape[0], im.shape[1]))

M_translation = np.float32([[1, 0, 100], [0, 1, 100]])
im_translation = cv2.warpAffine(im, M_translation, (im.shape[1], im.shape[0]))

fig, axes = plt.subplots(4,2,figsize=(15,30))

axes[0, 0].imshow(im, cmap='gray')
axes[0, 0].set_title("Source Image")
axes[0, 1].imshow(harris_corner_detector(im), cmap='gray')
axes[0, 1].set_title("harris corner")

axes[1, 0].imshow(im_scale, cmap='gray')
axes[1, 0].set_title("scaled Image")
axes[1, 1].imshow(harris_corner_detector(im_scale), cmap='gray')
axes[1, 1].set_title("scaled harris corner")

axes[2, 0].imshow(im_rotation, cmap='gray')
axes[2, 0].set_title("rotated Image")
axes[2, 1].imshow(harris_corner_detector(im_rotation), cmap='gray')
axes[2, 1].set_title("rotated harris corner")

axes[3, 0].imshow(im_translation, cmap='gray')
axes[3, 0].set_title("translation Image")
axes[3, 1].imshow(harris_corner_detector(im_translation), cmap='gray')
axes[3, 1].set_title("translation harris corner")

fig.tight_layout()
plt.show()
plt.savefig('./corner.jpg')
