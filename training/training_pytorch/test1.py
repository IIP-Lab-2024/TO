import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import torchvision.utils as vutils

npImg1 = cv2.imread("_040.png")
npImg2 = cv2.imread("_001.png")
img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

img1 = Variable(img1, requires_grad=False)
img2 = Variable(img2, requires_grad=True)

# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).item()  ###
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

epoch = 1   ###
while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = round(- ssim_out.item(), 4)  ### round 保留4位小数
    # save image
    vutils.save_image(img2, f'SSIMPNG/einstein_{epoch}_{ssim_value}.png')  ### 官网教程没有保存每一次迭代的图像，这里我们保存下来。 如果没有SSIMPNG这个文件夹，手动创建一个。
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()
    epoch += 1   ###