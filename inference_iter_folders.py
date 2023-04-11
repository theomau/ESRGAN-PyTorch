# -*- coding: utf-8 -*-
"""
Created on March 2023

@author: Theo
"""
import argparse
import os
import time
import cv2
import torch
from torch import nn
from tqdm import tqdm

import imgproc
import model
from utils import load_state_dict


model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    g_model = model.__dict__[model_arch_name](in_channels=3,
                                              out_channels=3,
                                              channels=64,
                                              growth_channels=32,
                                              num_blocks=23)
    g_model = g_model.to(device=device)

    return g_model


def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    g_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    g_model = load_state_dict(g_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")
    
 
    # Start the verification mode of the model.
    g_model.eval()
    
    for img_name in tqdm(os.listdir(args.input_dir)):
        
        time_start = time.time()

        # get image name
        img_path = f"{args.input_dir}/{img_name}"
        lr_tensor = imgproc.preprocess_one_image(img_path, device)
        
        # Use the model to generate super-resolved images
        with torch.no_grad():
            sr_tensor = g_model(lr_tensor)
            
        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        img_name = img_name.replace("LR", "SR")
        output_path = f"{args.output_dir}/{img_name}"
        cv2.imwrite(output_path, sr_image)
        time_end = time.time()
        
        print('\n time for generate SR : ',time_end-time_start)
        print(f"\n SR image save to `{output_path}`")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="rrdbnet_x2")
    parser.add_argument("--input_dir",
                        type=str,
                        default="./input_images/",
                        help="Path to folder containing low-resolution images.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output_images/",
                        help="Path to folder in which to save super-resolution images.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/train_ESRGAN_x2_div2k_Freeze_DG/g_last.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        raise ValueError("This is not a directory!")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    main(args)
    
    
    
