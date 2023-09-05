from image_quality_assessment import MSE, PSNR, SSIM, LPIPS_Score
import os
import cv2
import csv
import yaml
import torch
import argparse
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from utils import build_iqa_model

def save_metrics(hr, sr, psnr_model, ssim_model, lpips_model, device):
    hr_tensor = T.ToTensor()(hr)
    sr_tensor = T.ToTensor()(sr)
    hr_tensor = hr_tensor.to(device, non_blocking=True).reshape(1,3,32,32)
    sr_tensor = sr_tensor.to(device, non_blocking=True).reshape(1,3,32,32)
    
    psnr_score = psnr_model(sr_tensor, hr_tensor)
    ssim_score = ssim_model(sr_tensor, hr_tensor)

    LPIPS_score = lpips_model(sr_tensor, hr_tensor)
    
    return psnr_score.item() ,ssim_score.item(),LPIPS_score


def image_crop(hrimage,srimage, device:torch.device,
               psnr_model:nn.Module,
               ssim_model:nn.Module,
               lpips_model:nn.Module,
               patchsize = 32):
    range_w = (int)(hrimage.width/patchsize)
    range_h = (int)(hrimage.height/patchsize)
    result = []
    

    for w in range(range_w):
        for h in range(range_h):
            bbox = (w*patchsize, h*patchsize, (w+1)*patchsize, (h+1)*patchsize)
            crop_hrimg = hrimage.crop(bbox)
            crop_srimg = srimage.crop(bbox)

            crop_srimg = np.array(crop_srimg)
            crop_srimg = cv2.cvtColor(crop_srimg, cv2.COLOR_RGB2BGR)
            crop_hrimg = np.array(crop_hrimg)
            crop_hrimg = cv2.cvtColor(crop_hrimg,cv2.COLOR_RGB2BGR)
            psnr, ssim, lpips = save_metrics(crop_hrimg,crop_srimg,psnr_model,
                                             ssim_model,
                                             lpips_model,
                                             device)

            result.append([bbox, psnr, ssim, lpips])

    return result
    
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="./configs/patches/0710GAN_x4-DIV2K_mg109_1.yaml",
                        required=True,
                        help="Path to test config file.")
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)
        
    device = torch.device("cuda", config["DEVICE_ID"])
    #define iqa model
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["ONLY_TEST_Y_CHANNEL"],
        device
    )
    lpips_model = LPIPS_Score("vgg", "0.1")
    lpips_model = lpips_model.to(device)
    
    save_path = os.path.join(config["SAVE_DIR_PATH"].format(EXP_NAME = config["EXP_NAME"],
                                                            DATASETNAME = config["DATASETNAME"]))
    #sr data, gt data path
    sr_path = config["SR_DIR_PATH"].format(EXP_NAME = config["EXP_NAME"],
                                                            DATASETNAME = config["DATASETNAME"])
    hr_path = config["GT_DIR_PATH"].format(EXP_NAME = config["EXP_NAME"],
                                                            DATASETNAME = config["DATASETNAME"])
    
    with open(save_path,"a", newline = '') as f:
        wr = csv.writer(f) 
        wr.writerow(["dataset","imagename","grid","psnr","ssim","lpips"])
        
        #patch crop
        sr_list = sorted(os.listdir(sr_path))
        import pdb; pdb.set_trace()
        for idx, file in enumerate(sr_list):
            if idx >20:
                pass
            else:
                srimgpath = os.path.join(sr_path,file)
                hrimgpath = os.path.join(hr_path,file)
                imagename = file
                
                srimg = Image.open(srimgpath)
                hrimg = Image.open(hrimgpath)
                
                result = image_crop(hrimg,srimg,device, psnr_model, ssim_model,
                                    lpips_model)
                
                for patchresult in result:
                    grid = patchresult[0]
                    psnr = patchresult[1]
                    ssim = patchresult[2]
                    lpips = patchresult[3]
                    wr.writerow([config["DATASETNAME"],imagename,
                                grid, psnr,ssim,lpips])
                print(imagename,"done")
if __name__ == '__main__':
    main()   
    














