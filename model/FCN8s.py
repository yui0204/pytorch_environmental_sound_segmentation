import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from .BasicModule import BasicModule

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class FCN8s(BasicModule):
    def __init__(self, n_classes=75, input_dim=1):
        super().__init__()
        self.model_name = 'FCN8s'
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        self.upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 16, stride=8)
        self.upscore4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2)
        self.upscore5 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score5 = self.classifier(conv5)
        
        upscore5 = self.upscore5(score5)
        score4 = self.score_pool4(conv4)
        score4 = score4[:, :, 5:5+upscore5.size()[2], 5:5+upscore5.size()[3]].contiguous()
        score4 += upscore5

        upscore4 = self.upscore4(score4)
        score3 = self.score_pool3(conv3)
        score3 = score3[:, :, 9:9+upscore4.size()[2], 9:9+upscore4.size()[3]].contiguous()
        score3 += upscore4

        mask = self.upscore(score3)
        mask = mask[:, :, 31:31+x.size()[2], 31:31+x.size()[3]].contiguous()
        mask = torch.sigmoid(mask)
        out = torch.mul(x, mask)

        return out
