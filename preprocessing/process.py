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


def preprocess(list_images, resize, to_gray, normalize, denoise, sharpen):
    processed_imgs = []
    for img in list_images:
        if resize:
             img = cv2.resize(img, 256, 256)
        if to_gray:           
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

        if normalize: 
            img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if denoise:
            img = filter(img)

        if sharpen:
            img = filter(img, 'sharpen')

        processed_imgs.append(img)
        break
    return processed_imgs


