# AllConv model for automatic piano transcription
# Based on: Kelz et al., "On the Potential of Simple Framewise Approaches to Piano Transcription"
# https://arxiv.org/pdf/1612.05153

import torch
import torch.nn as nn
import torch.nn.functional as F

class AllConv(nn.Module):
    def __init__(self):
        super(AllConv, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(1, 25), padding=(0, 0))
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(1, 25), padding=(0, 0))
        self.conv7 = nn.Conv2d(128, 88, kernel_size=(1, 1), padding=(0, 0))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(88)

        self.max_pool = nn.MaxPool2d((1, 2))
        self.avg_pool = nn.AvgPool2d((1, 6))

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, 1, 5, 229)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.max_pool(x)
        x = self.drop2(x)

        x = F.relu(self.conv5(x))
        x = self.bn4(x)
        x = F.relu(self.conv6(x))
        x = self.bn5(x)
        x = self.drop3(x)

        x = self.conv7(x)
        x = self.bn6(x)
        x = self.avg_pool(x)

        # Model output: (batch, 88, 5, 1)
        x = x[:, :, 2, :] # Take centre frame (keep middle slice)
        x = x.permute(0, 2, 1)

        return x
