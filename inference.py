# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import time
import cv2
import torch
from torch import nn
import torchsummary
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
    #print(g_model)
    
    
    # Load model weights
    g_model = load_state_dict(g_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")
    
    #torchsummary.summary(g_model,input_size=(3,128,128))
   
    # Uncomet when we want Freeze the layer
    g_model.conv1.requires_grad_(False)
    g_model.trunk.requires_grad_(False)
    g_model.conv2.requires_grad_(False)
    
    print("**********************")
    for name, param in g_model.named_parameters():
        if param.requires_grad == True:
            print (name)
    print("**********************")
    
    # Start the verification mode of the model.
    g_model.eval()
    
    for name, param in g_model.named_parameters():
        if param.requires_grad == True:
            print (name)
    
    

    time_start = time.time()
    lr_tensor = imgproc.preprocess_one_image(args.inputs_path, device)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = g_model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, sr_image)
    time_end = time.time()
    print('time for generate SR : ',time_end-time_start)
    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="rrdbnet_x2")
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure/Diamond_LR1.png",
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure/Diamond_SR1bb.png",
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/g_epoch_30.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
