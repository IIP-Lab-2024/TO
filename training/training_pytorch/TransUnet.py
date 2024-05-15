import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import Transformer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(TransUNet, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Transformer
        self.transformer = Transformer(
            d_model=256, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4)
        self.conv4 = ConvBlock(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        x = self.pool3(x3)

        # Transformer
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1) # Reshape for transformer
        x = self.transformer(x, x) # Pass the same tensor as src and tgt
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Reshape back to image format

        # Decoder
        x = self.upconv3(x)
        x = torch.cat([x, x2[:, :, :x.shape[2], :x.shape[3]]], dim=1)  # Add slicing to match the dimensions
        x = self.conv4(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1[:, :, :x.shape[2], :x.shape[3]]], dim=1)  # Add slicing to match the dimensions
        x = self.conv5(x)
        x = self.final_conv(x)

        return x



# Test the model with

# Create the model and input tensor
model = TransUNet(in_channels=1, out_channels=1, num_heads=8, num_layers=6)
input_data = torch.randn(128, 1, 64, 128)
output = model(input_data)
print(output.shape)