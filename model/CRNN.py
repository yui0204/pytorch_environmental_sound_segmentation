import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from .BasicModule import BasicModule


class CRNN(BasicModule):
    def __init__(self, n_classes=75, angular_resolution=1, input_dim=1):
        super().__init__()
        self.model_name = 'CRNN'
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.gru_block = nn.Sequential(
            nn.GRU(512, 512, 2, bidirectional=False),
            #nn.LSTM(512, 512, 2, bidirectional=False),
        )

        
        self.deconv_block5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, (2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv_block0 = nn.Sequential(
            nn.ConvTranspose2d(128, self.n_classes * angular_resolution, (2, 1), stride=(2, 1)),
        )


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        conv6 = self.conv_block6(conv5)
        
        gru = conv6.view(-1, 512, 4*256)
        gru = gru.transpose(1, 2)
        gru, (hn, cn) = self.gru_block(gru)
        gru = gru.transpose(1, 2)
        gru = gru.view(-1, 512, 4, 256)
        
        deconv5 = self.deconv_block5(gru)
        deconv5 = torch.cat((conv5, deconv5), 1)
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
        out = torch.mul(x[:, 0, :, :].unsqueeze(1), mask)

        return out
