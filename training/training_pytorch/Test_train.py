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
from DoubleUnet import build_doubleunet
from CBA_Unet import CBA_UNet
import torch
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
num_epochs = 50


class CustomLoss(nn.Module):
    def __init__(self, alpha):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # 计算 MAE 和 MSE
        loss_mae = F.l1_loss(y_pred, y_true)
        loss_mse = F.mse_loss(y_pred, y_true)

        # 计算加权平均损失
        loss = self.alpha * loss_mae + (1 - self.alpha) * loss_mse
        return loss

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
with np.load('data/random_input_data.npz') as data:
    input_data = data['input']
with np.load('data/random_target_data.npz') as data:
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
#model = UNet().to(device)
#model =build_doubleunet().to(device)
model =CBA_UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.01)

# 定义损失函数
#criterion = torch.nn.L1Loss().to(device)
#criterion = torch.nn.MSELoss().to(device)
#criterion = CustomLoss(alpha=0.5).to(device)
criterion= SSIM(window_size = 11)
# 训练模型
# Train the model
def training():
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    train_loss = []
    val_loss = []
    lr_list, epoch_list = list(), list()
    total_step = 0
    best_acc, best_epoch = 0, 0
    print("------train start------")
    print()
    for epoch in range(num_epochs):
        t0 = time.time()
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        model.train()
        for i, (x, y) in enumerate(train_loader):
            # 将数据放到 GPU 上（如果可用）
            inputs = x.to(device)
            labels = y.to(device)
            # 前向传播
            outputs = model(inputs)
            loss = -criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step += 1

        train_loss.append((1+loss.item()))
        t1 = time.time()
        scheduler.step()

            # 打印统计信息
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, (1+loss.item())))
        print('Train Epoch time: ', (t1 - t0))
        # 保存模型的检查点
      #  torch.save(model.state_dict(), './model/cantilever_model/cantilever_best_32_64.ckpt')

        print("------valid start------")
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            iouacc = 0
            iouacc1 = 0
            for x, y in test_loader:
                inputs = x.to(device)
                labels = y.to(device)
                output = model(inputs).to(device)
                valloss = criterion(output, labels)
                acc, acc1 = iou(output, y.to(device))  # .detach().cpu()
                iouacc += acc
                iouacc1 += acc1
                # 二值化
                th = 0.5
                predicted = output.data
                pred_data = predicted
                pred_data[predicted > th] = 1.0
                pred_data[predicted <= th] = 0.001
                total += labels.size(0)
                correct += (pred_data == labels).sum().item()
            val_loss.append((1-valloss.item()))
            print('Acc: {} %'.format(100 * (correct / (total * 128 * 64))))
            print("Val IOU:", iouacc / len(test_loader))
            print("IOU>0.9:", iouacc1 / len(test_set))
            print('Valloss: {:.4f}'.format((1-valloss.item())))

        val_acc = iouacc / len(test_loader)
        if val_acc >= best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), './model/simply_model/simply_best_64_128.ckpt')
        print('best acc:', best_acc, 'best epoch:', best_epoch)
        print()



def iou(pred, target):
    count=0
    pred = (pred > 0.5).int()    # 将浮点数转换为0或1的整数
    target = target.int()        # 确保目标值是整数
    intersection = (pred & target).float().sum((2, 3))
    union = (pred | target).float().sum((2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    count += (iou > 0.90).sum().item()

    return iou.mean(),count


def test():
    # load model
    weight = torch.load('./model/simply_model/simply_best_64_128.ckpt')
    model.load_state_dict(weight)
    print("------test start best------")
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
            acc,acc1 = iou(output, y.to(device))#.detach().cpu()
            iouacc+=acc
            iouacc1 += acc1
            #二值化
            th=0.5
            predicted = output.data
            pred_data=predicted
            pred_data[predicted>th]=1.0
            pred_data[predicted<=th]=0.001
            total += labels.size(0)
            correct += (pred_data == labels).sum().item()
        print('Acc: {} %'.format(100 * (correct / (total * 128*64))))
        print("Test IOU:", iouacc / len(test_loader))
        print("IOU>0.9:", iouacc1 / len(test_set))


if __name__ == '__main__':
    training()
    test()
    print("over")


