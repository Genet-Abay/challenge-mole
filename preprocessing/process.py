import cv2
import numpy as np
import sklearn as sns
from dataacquisition import read_data
from matplotlib import pyplot as plt


def filter(img, type='denoising'):
    if type=='denoising':
        filtered_img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)

    else:# if not denoising, sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        filtered_img = cv2.filter2D(img, -1, kernel)

    return filtered_img


def preprocess(list_images, ):
    processed_imgs = []
    for img in list_images:
        img = cv2.resize(img, 256, 256)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       

        # img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = filter(img)
        processed_imgs.append(img)
        break
    return processed_imgs


