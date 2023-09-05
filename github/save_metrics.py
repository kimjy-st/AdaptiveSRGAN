from image_quality_assessment import MSE, PSNR, SSIM, LPIPS_Score
import os
import csv
from PIL import Image
from utils import build_iqa_model
import torchvision.transforms as T

def save_metrics(hr, sr, psnr_model, ssim_model, lpips_model, device):
    hr_tensor = T.ToTensor()(hr)
    sr_tensor = T.ToTensor()(sr)
    hr_tensor = hr_tensor.to(device, non_blocking=True)
    sr_tensor = sr_tensor.to(device, non_blocking=True)
    hr_tensor = hr_tensor.unsqueeze(0)
    sr_tensor = sr_tensor.unsqueeze(0)

    psnr_score = psnr_model(sr_tensor, hr_tensor)
    ssim_score = ssim_model(sr_tensor, hr_tensor)

    LPIPS_score = lpips_model(sr_tensor, hr_tensor)
    
    return psnr_score.item() ,ssim_score.item(),LPIPS_score

def get_metrics(hrdirroot, srdirroot, savedirroot):
    results = []
    for file in os.listdir(srdirroot):
        imagename = file 

        srimageroot = os.path.join(srdirroot, imagename)
        hrimageroot = os.path.join(hrdirroot, imagename)

        hrimageroot = Image.open(hrimageroot)
        srimageroot = Image.open(srimageroot)
        
        psnr_model, ssim_model = build_iqa_model(4, True, "cuda:3")
        lpips_model = LPIPS_Score("vgg","0.1")
        lpips_model = lpips_model.to("cuda:3")
        
        psnr_score, ssim_score,LPIPS_score = save_metrics(hrimageroot, srimageroot,
                                                          psnr_model, ssim_model, lpips_model, "cuda:3")
    
        result = [imagename, psnr_score ,ssim_score,LPIPS_score]
        results.append(result)
        l = len(os.listdir(srdirroot))
        print("Get metrics!  remain : ",l-len(results))

    listname = ["imagename", "PSNR", "SSIM", "LPIPS"]
    with open(savedirroot,"a",newline = '') as f:
        wr = csv.writer(f)
        wr.writerow(listname) 

    for i in range(len(results)): 
        with open(savedirroot,"a",newline = '') as f:
            wr = csv.writer(f)
            wr.writerow(results[i])


foldername = "0721GAN_x4-DIV2K_ms_8000"
""" foldername:
                "ASRGAN_x4-DIV2K-1000-buffer" or "EDSR-PyTorch" or "SRGAN_x4-ImageNet-BASE"or SRResNet """
                
SRGAN_sr = [f"./{foldername}/Set5",f"./{foldername}/Set14",f"./{foldername}/BSDS100",f"./{foldername}/Manga109",f"./{foldername}/Urban100"]
EDSR = [f"../{foldername}/experiment/test/Set5",f"../{foldername}/experiment/test/Set14",f"../{foldername}/experiment/test/BSDS100",
        f"../{foldername}/experiment/test/Manga109",f"../{foldername}/experiment/test/Urban100"]
GT_dirs = ["/raid/datasets/SR_dataset/Set5/GTmod12","/raid/datasets/SR_dataset/Set14/GTmod12",
           "/raid/datasets/SR_dataset/BSDS100/GTmod12","/raid/datasets/SR_dataset/Manga109/GTmod12",
           "/raid/datasets/SR_dataset/Urban100/GTmod12"]

for i in range(len(SRGAN_sr)):
    gtdirroot = GT_dirs[i] 
    srdirroot = SRGAN_sr[i]
    datasetname = SRGAN_sr[i].split("/")[-1]
    savedirroot = f"./{foldername}/{foldername}_{datasetname}result.csv"

    get_metrics(gtdirroot, srdirroot, savedirroot)














