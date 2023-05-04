# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:09:31 2023

@author: Theo
"""
import cv2
from skimage import io

image_path = "D://Théo//ESRGAN-PyTorch//output_images//"

sr_46 = io.imread(image_path + "46_SR.tif")
io.imshow(sr_46)
io.show()