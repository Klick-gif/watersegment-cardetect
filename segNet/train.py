import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt




class SegNet(nn.Module):
    """SegNet模型实现"""
    def __init__(self, num_classes=1, in_channels=3):
        super(SegNet, self).__init__()
        
        # Encoder
        self.enc1 = self._encoder_block(in_channels, 64, 2)
        self.enc2 = self._encoder_block(64, 128, 2)
        self.enc3 = self._encoder_block(128, 256, 3)
        self.enc4 = self._encoder_block(256, 512, 3)
        self.enc5 = self._encoder_block(512, 512, 3)
        
        # Decoder
        self.dec5 = self._decoder_block(512, 512, 3)
        self.dec4 = self._decoder_block(512, 256, 3)
        self.dec3 = self._decoder_block(256, 128, 3)
        self.dec2 = self._decoder_block(128, 64, 2)
        self.dec1 = self._decoder_block(64, num_classes, 2)
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
    def _encoder_block(self, in_ch, out_ch, num_conv):
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_ch, out_ch, num_conv):
        layers = []
        for i in range(num_conv):
            conv_in = in_ch if i == 0 else out_ch
            layers.append(nn.Conv2d(conv_in, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            if i < num_conv - 1:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1_pool, idx1 = self.pool(e1)
        
        e2 = self.enc2(e1_pool)
        e2_pool, idx2 = self.pool(e2)
        
        e3 = self.enc3(e2_pool)
        e3_pool, idx3 = self.pool(e3)
        
        e4 = self.enc4(e3_pool)
        e4_pool, idx4 = self.pool(e4)
        
        e5 = self.enc5(e4_pool)
        e5_pool, idx5 = self.pool(e5)
        
        # Decoder
        d5 = self.unpool(e5_pool, idx5, output_size=e5.size())
        d5 = self.dec5(d5)
        
        d4 = self.unpool(d5, idx4, output_size=e4.size())
        d4 = self.dec4(d4)
        
        d3 = self.unpool(d4, idx3, output_size=e3.size())
        d3 = self.dec3(d3)
        
        d2 = self.unpool(d3, idx2, output_size=e2.size())
        d2 = self.dec2(d2)
        
        d1 = self.unpool(d2, idx1, output_size=e1.size())
        d1 = self.dec1(d1)
        
        return d1


