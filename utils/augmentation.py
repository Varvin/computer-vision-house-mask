import cv2
import numpy as np

def blur(image):
    sigmaX = np.random.rand()
    image_ = cv2.GaussianBlur(image,(5,5),sigmaX)
    return image_

def flip(image, mask):
    image_ = np.fliplr(image)
    mask_ = np.fliplr(mask)
    return image_, mask_

def rotate(image, mask):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    angle = np.random.randint(1,25)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    image_ = cv2.warpAffine(image, M, (w, h))
    mask_ =  cv2.warpAffine(mask, M, (w, h))
    return image_, mask_