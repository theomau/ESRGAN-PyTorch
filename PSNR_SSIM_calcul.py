# -*- coding: utf-8 -*-
"""
Created on Apr 2023

@author: Theo
"""
import os
import argparse
import cv2
import pandas as pd
from image_quality_assessment import psnr, ssim
from tqdm import tqdm


import cvxpy as cp

def total_variation(arr):
    dx = cp.vec(arr[1:, :-1] - arr[:-1, :-1])
    dy = cp.vec(arr[:-1, 1:] - arr[:-1, :-1])
    D = cp.vstack((dx, dy))
    norm = cp.norm(D, p=1, axis=0)
    return cp.sum(norm)



def main(arsgs) -> None:
    
    # initialisation of list
    image_SR_name = []
    psnr_y_channel = []
    psnr_value = []
    ssim_y_channel = []
    ssim_value = []
    #total_variation = []
    

    
    for image_name in tqdm(os.listdir(args.sr_dir)):
        
        # read path and open image
        image_sr_path = f"{args.sr_dir}/{image_name}"
        image_hr_path = f"{args.hr_dir}/{image_name}"
        image_hr_path = image_hr_path.replace('SR', 'HR')
        
        image_hr = cv2.imread(image_hr_path)
        image_sr = cv2.imread(image_sr_path)
        
        
        # write value in list
        image_SR_name.append(image_name)
        psnr_y_channel.append( psnr(image_hr, image_sr, crop_border = 0, only_test_y_channel = True) )
        psnr_value.append( psnr(image_hr, image_sr, crop_border = 0, only_test_y_channel = False) )
        ssim_y_channel.append( ssim(image_hr, image_sr, crop_border = 0, only_test_y_channel = True) )
        ssim_value.append( ssim(image_hr, image_sr, crop_border = 0, only_test_y_channel = False) )
      
        
    # create a dictionary
    dict = {'Image name' : image_SR_name,
            'PSNR y channel' : psnr_y_channel,
            'PSNR' : psnr_value,
            'SSIM y channel' : ssim_y_channel,
            'SSIM' : ssim_value,
            #'Total Variation' : total_variation
            }
       
    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(dict) 
    
    # write the DataFrame to a CSV file
    df.to_csv(f"{args.name_experience}.csv") 
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using for calculat psnr and ssim of images SR generation.")
    
    parser.add_argument("--name_experience",
                        type=str,
                        default="result_experience",
                        help="Name of experince test.")
    parser.add_argument("--hr_dir",
                        type=str,
                        default="./hr_images/",
                        help="Path to folder containing hight-resolution images.")
    parser.add_argument("--sr_dir",
                        type=str,
                        default="./output_images/",
                        help="Path to folder containing super-resolution images.")
    
    args = parser.parse_args()
    
    main(args)