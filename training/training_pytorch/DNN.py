import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFeedForwardNN(nn.Module):
    def __init__(self):
        super(DeepFeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()
        input_features = 1*32*64
        self.fc1 = nn.Linear(input_features, 1024)  # 第一层
        self.bn1 = nn.BatchNorm1d(1024)              # 批量归一化
        self.dp1 = nn.Dropout(0.5)                   # 丢弃层
        self.fc2 = nn.Linear(1024, 512)              # 第二层
        self.bn2 = nn.BatchNorm1d(512)               # 批量归一化
        self.dp2 = nn.Dropout(0.5)                   # 丢弃层
        self.fc3 = nn.Linear(512, 256)               # 第三层
        self.bn3 = nn.BatchNorm1d(256)               # 批量归一化
        self.fc4 = nn.Linear(256, input_features)    # 输出层，尺寸与输入相同

    def forward(self, x):
        x = self.flatten(x)
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = x.view(-1, 1, 32, 64)  # 重新调整尺寸以匹配输出尺寸
        return x

# 模型实例化
model = DeepFeedForwardNN()

# 假设有一个输入
input_tensor = torch.randn(128, 1, 32, 64)  # 批量大小为128的示例输入

# 通过模型前向传递输入
output_tensor = model(input_tensor)

print(output_tensor.size())  # 输出尺寸应该是 torch.Size([128, 1, 32, 64])
