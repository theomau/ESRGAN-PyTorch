# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:24:09 2023

@author: Theo
"""
import random

import numpy as np
import torch
from torch.backends import cudnn


# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
g_arch_name = "g_p4_rrdbnet_x2"
# Model arch config
in_channels = 3
out_channels = 3
channels = 64
growth_channels = 32
num_blocks = 2
upscale_factor = 2
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "train_g_p4_RRDBNet_x2_Diamond_no_DA"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/Diamond_q2_no_contrast/Train_Diamond2_HR"

    test_gt_images_dir =  f"./data/Diamond_q2_no_contrast/Test_Diamond2_HR"
    
    test_lr_images_dir =  f"./data/Diamond_q2_no_contrast/Test_Diamond2_LR"

    gt_image_size = 128
    batch_size = 6 # nombres d'image par lots
    num_workers = 4 # le nombres de sous-procésseurs utilisé pour charger les données dans la fonction DataLoader

    # The address to load the pretrained model
    pretrained_g_model_weights_path = f"./results/pretrained_models/g_last_ESRGAN_x4.pth.tar"

    # Incremental training and migration training
    resume_g_model_weights_path = f""

    # Total num epochs 
    epochs = 90

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/Diamond_q2_no_contrast/Test_Diamond2_LR"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = f"./data/Diamond_q2_no_contrast/Test_Diamond2_HR"
    
