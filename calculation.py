# This py file  is used for calculating psnr and ssim of dataset
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# read the folder of two files： one is the groud thurth, the other is restored one.
ground_dir = 'E:/Derain_AGUNetcode/rain12/target'
resotred_dir  = 'E:/2014-Publication/2022 Gradient domain knowledge driven deep learning deraining\method_comparison/rain12/our/recovered'

data_ground = os.listdir(ground_dir)
data_restored = os.listdir(resotred_dir)

psnr_values = []
ssim_values = []

for idx, (data_ground,data_restored) in enumerate (zip(data_ground,data_restored),start=1):
    img1 = cv2.imread(os.path.join(ground_dir,data_ground))
    img2 = cv2.imread(os.path.join(resotred_dir,data_restored))

    # transform the color images into gray images
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # 计算PSNR
    psnr_val = psnr(gray1,gray2)
    psnr_values.append(psnr_val)

    # 计算SSIM
    ssim_val,_ = ssim(gray1,gray2,full=True)
    ssim_values.append(ssim_val)

    print(f"processing pair {idx}:{data_ground} and {data_restored}")

# 计算平均值
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR:{avg_psnr}, Average SSIM:{avg_ssim }")