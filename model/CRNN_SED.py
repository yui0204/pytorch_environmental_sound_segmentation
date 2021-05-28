import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from .BasicModule import BasicModule


class CRNN_SED(BasicModule):
    def __init__(self, n_classes=75, angular_resolution=1, input_dim=1):
        super().__init__()
        self.model_name = 'CRNN_SED'
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.gru_block = nn.Sequential(
            nn.GRU(512, 512, 2, bidirectional=True),
            #nn.LSTM(512, 512, 2, bidirectional=False),
        )

        self.event_fc = nn.Linear(1024, self.n_classes * angular_resolution, bias=True)


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.conv_block8(conv7)
        
        gru = conv8.view(-1, 512, 256)
        gru = gru.transpose(1, 2)
        (gru, _) = self.gru_block(gru)
        
        out = torch.sigmoid(self.event_fc(gru))
        out = out.transpose(1, 2)
        
        return out
