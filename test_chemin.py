# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:50:30 2023

@author: Theo
"""

import os

image_dir_hr = 'D:/Théo/ESRGAN-PyTorch/data/Train_Diamond2_HR'
image_dir_lr = image_dir_hr.replace('HR', 'LR')

image_file_names_hr = [os.path.join(image_dir_hr, image_file_name) for image_file_name in os.listdir(image_dir_hr)]
image_file_names_lr = [path.replace('HR', 'LR') for path in image_file_names_hr]
