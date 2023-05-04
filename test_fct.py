# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:07:24 2023

@author: Theo
"""
import cv2
import numpy as np
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def TV(image) -> float:
    """Calcul the total variation for greyscale image
    Args: image (np.ndarray)
    Returns: the value of total variation
    
    """
    n, m = image.shape
    som = 0
    for i in range(n):
        for j in range(m):
            som = som + np.sqrt( (image[i-1,j-1]-image[i,j-1])**2 + (image[i-1,j-1]-image[i-1,j])**2)
    return som


def TV_norm1(image_hr, image_sr) -> float:
    """Calcul the difference of total variation for greyscale image for norm 1
    Args: image (np.ndarray)
    Returns: the value of total variation
    
    """
    n, m = image_hr.shape
    return 1/(n*m) * np.absolute( TV(image_hr) - TV(image_sr) )/255


def TV_norm2(image_hr, image_sr) -> float:
    """Calcul the difference of total variation for greyscale image for norm 2
    Args: image (np.ndarray)
    Returns: the value of total variation
    
    """
    n, m = image_hr.shape
    som = 0
    for i in range(n):
        for j in range(m):
            som = som + np.sqrt( np.absolute( image_hr[i-1,j-1]-image_hr[i,j-1]-image_sr[i-1,j-1]-image_sr[i,j-1])**2 + np.absolute(image_hr[i-1,j-1]-image_hr[i-1,j] -image_sr[i-1,j-1]-image_sr[i-1,j])**2)
    return 1/(n*m) * som / 255

image_hr_path = "./hr_images/46_HR.tif"
image_sr_path = "./output_images/46_SR.tif"
image_hr = cv2.imread(image_hr_path, 0)
image_sr = cv2.imread(image_sr_path, 0)

TV_norm1(image_hr, image_sr)
