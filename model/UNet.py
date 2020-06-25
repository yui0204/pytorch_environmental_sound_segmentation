import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from .BasicModule import BasicModule


class UNet(BasicModule):
    def __init__(self, n_classes=75, input_dim=1):
        super().__init__()
        self.model_name = 'UNet'
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


        self.deconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv_block0 = nn.Sequential(
            nn.ConvTranspose2d(128, self.n_classes, 2, stride=2),
        )


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        
        deconv4 = self.deconv_block4(conv5)
        deconv4 = torch.cat((conv4, deconv4), 1)
        deconv3 = self.deconv_block3(deconv4)
        deconv3 = torch.cat((conv3, deconv3), 1)
        deconv2 = self.deconv_block2(deconv3)
        deconv2 = torch.cat((conv2, deconv2), 1)
        deconv1 = self.deconv_block1(deconv2)
        deconv1 = torch.cat((conv1, deconv1), 1)
        deconv0 = self.deconv_block0(deconv1)

        mask = torch.sigmoid(deconv0)
        out = torch.mul(x, mask)

        return out
