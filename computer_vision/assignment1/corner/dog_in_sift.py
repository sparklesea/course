import numpy as np
import cv2

im = cv2.imread('./chessboard.jpg', 0)
im_scale = cv2.resize(im, None, fx=0.5, fy=0.5)

M_rotation = cv2.getRotationMatrix2D((im.shape[0] / 2, im.shape[1] / 2), 45, 1)
im_rotation = cv2.warpAffine(im, M_rotation, (im.shape[0], im.shape[1]))

M_translation = np.float32([[1, 0, 100], [0, 1, 100]])
im_translation = cv2.warpAffine(im, M_translation, (im.shape[1], im.shape[0]))

sift = cv2.SIFT_create(contrastThreshold=0.09)


def dog_in_sift(im):
    gauss1 = cv2.GaussianBlur(im, (5, 5), 1.2)
    gauss2 = cv2.GaussianBlur(im, (5, 5), 1.3)
    DoG = gauss1 - gauss2  # DoG算子

    interestpoint = sift.detect(im, DoG)
    corner = cv2.drawKeypoints(im, interestpoint, im, (0, 0, 255))
    return corner


corner = dog_in_sift(im)
corner_scale = dog_in_sift(im_scale)
corner_rotation = dog_in_sift(im_rotation)
corner_translation = dog_in_sift(im_translation)

cv2.imwrite('./sift_corner.jpg', corner)
cv2.imwrite('./sift_corner_scale.jpg', corner_scale)
cv2.imwrite('./sift_corner_rotation.jpg', corner_rotation)
cv2.imwrite('./sift_corner_translation.jpg', corner_translation)
