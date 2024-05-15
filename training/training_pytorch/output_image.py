from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import StepLR
from unet import UNet
from SSIM_loss import SSIM
from pre_training_net import preUNet
from DoubleUnet2 import build_doubleunet
import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
torch.cuda.is_available()
print(torch.cuda.device_count())
device = torch.device('cuda:1')

# Define some hyperparameters
batch_size = 128
learning_rate = 0.01

class CustomDataset(Dataset):
    def __init__(self, input_data, target_data, transform=None):
        self.input_data = input_data.astype(np.float32)
        self.target_data = target_data.astype(np.float32)
        self.transform = transform
        self.transform1 = transform1
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]
        x = to_pil_image(x)
        y = to_pil_image(y)
        if self.transform:
            x = self.transform(x)
            x = self.transform1(x)
            y = self.transform(y)
        return x, y

# Define data transformations for data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
])
transform1 = transforms.Compose([
    transforms.Normalize((0.2,), (0.2,)),
])

# Load the input and target data
with np.load('data/simply_supported_beam_input_64_128.npz') as data:
    input_data = data['input']
with np.load('data/simply_supported_beam_target_64_128.npz') as data:
    target_data = data['target']

dataset = CustomDataset(input_data, target_data,transform=transform)
dataset_size = len(dataset)
print("dataset_size:",dataset_size)

train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 0.1

train_size = int(train_ratio * dataset_size)
valid_size = int(valid_ratio * dataset_size)
test_size  = int(test_ratio * dataset_size)

torch.manual_seed(1)  # 设置随机种子，以便获得可重现的结果
train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
print('train_set:', len(train_set), '   val_set:', len(valid_set),'   test_set:', len(test_set))

# Create data loaders for training and validation and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=1)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True,num_workers=1)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers=1)


# Create a model and optimizer
#model = CNN().double()
model = build_doubleunet().to(device)
#model =build_doubleunet().to(device)


def iou(input ,pred, target):
    count=0
    iou =0
    pred = (pred > 0.5).int()    # 将浮点数转换为0或1的整数
    target = target.int()        # 确保目标值是整数
    for img_idx, (inputs,output,target) in enumerate(zip(input ,pred, target)):
        intersection = (output & target).float().sum((1, 2))
        union = (output | target).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou =iou.cpu().numpy()
        count += (iou > 0.90).sum().item()
        inputs = inputs.squeeze(1).permute(1, 2, 0).cpu().numpy()
        output = output.squeeze(1).permute(1, 2, 0).cpu().numpy()
        target = target.squeeze(1).permute(1, 2, 0).cpu().numpy()
        # 显示图片
        plt.imshow(inputs, cmap='Greys')
        plt.savefig(f'output/simply_64_128/{img_idx}_input_image_gaussian_{iou}.png')
        plt.close()
        plt.imshow(output, cmap='Greys')
        plt.savefig(f'output/simply_64_128/{img_idx}_output_image_gaussian_{iou}.png')
        plt.close()
        plt.imshow(target, cmap='Greys')
        plt.savefig(f'output/simply_64_128/{img_idx}_target_image_gaussian_{iou}.png')
        plt.close()
    return iou.mean(),count


def test():
    # load model
    weight = torch.load('./model/simply_model/simply_DoubleUnet2_best_64_128_SSIM.ckpt')
    model.load_state_dict(weight)
    print("------test start------")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        iouacc=0
        iouacc1=0
        for x, y in test_loader:
            inputs = x.to(device)
            labels = y.to(device)
            output = model(inputs).to(device)
            acc,acc1 = iou(inputs,output, labels)#.detach().cpu()
            iouacc+=acc
            iouacc1 += acc1
            #二值化
            th=0.5
            predicted = output.data
            pred_data=predicted
            pred_data[predicted>th]=1.0
            pred_data[predicted<=th]=0.001
            total += labels.size(0)
            break

        # print('Acc: {} %'.format(100 * (correct / (total *128*64))))
        # print("Test IOU:", iouacc / len(test_loader))
        # print("IOU>0.9:", iouacc1 / len(test_set))


if __name__ == '__main__':
    test()
    print("over")


