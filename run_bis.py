# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:46:41 2023

@author: Theo
"""

import os

# Prepare dataset
#os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/DIV2K_train_HR --output_dir ../data/DIV2K/ESRGAN/train --image_size 544 --step 272 --num_workers 16")
#os.system("python ./prepare_dataset.py --images_dir ../data/DIV2K/original/DIV2K_valid_HR --output_dir ../data/DIV2K/ESRGAN/valid --image_size 544 --step 544 --num_workers 16")


os.system(" python inference.py --model_arch_name rrdbnet_x2 --inputs_path ./figure/Diamond_LR20.png --output_path ./figure/Diamond_SR20ter.png --model_weights_path ./samples/train_RRDBNet_x2_div2k/g_epoch_16.pth.tar --device_type cuda")





#python inference.py --model_arch_name rrdbnet_x2 --input_dir ./input_images --output_dir ./output_images --model_weights_path ./samples/train_RRDBNet_x2_div2k/g_epoch_16.pth.tar --device_type cudapython inference.py --model_arch_name rrdbnet_x2 --inputs_path ./figure/Diamond_LR20.png --output_path ./figure/Diamond_SR20ter.png --model_weights_path ./samples/train_RRDBNet_x2_div2k/g_epoch_16.pth.tar --device_type cuda