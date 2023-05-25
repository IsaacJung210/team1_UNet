#!/usr/bin/env python3

# -*- coding:utf-8 -*-

import os, cv2
from glob import glob
import numpy as np

import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms
from PIL import Image

# import matplotlib.pyplot as plt

# 다운샘플링 4번 -> 각각 (컨벌루션,배치정규화,ReLU) 2번 / 리스트.append / 맥스풀링(2)1번
# 업샘플링 4번 -> 각각 (컨벌루션,배치정규화,ReLU) 2번 업컨벌루션(2x2)1번
# 최종 클래스 매핑 -> (컨벌루션,배치정규화,ReLU) 2번 컨벌루션(1x1)1번

class Unet(nn.Module):
    def __init__(self, class_num_:int, depth_:int, image_ch_:int, target_ch_:int=None):
        super(Unet,self).__init__()
        self.image_ch = image_ch_
        self.class_num = class_num_
        self.target_ch = target_ch_
        self.depth = depth_
        self.module_list = nn.ModuleList()
        self.samplings = nn.ModuleList()
        self.contract_path_mats = []
        self.dec_ch = 2 ** (self.depth + 1) * 64
        self.init_network()

        self.init_weights()

        print("  **  Unet init complete  **  \n")

    
    def init_weights(self):
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None : nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.ConvTranspose2d): 
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None : nn.init.constant_(m.bias, 0)


    def encoder(self, turn:int):
        self.mat = self.module_list[turn-1](self.mat)
        self.contract_path_mats.append(self.mat)      
        self.mat = self.samplings[turn-1](self.mat)

        if turn < self.depth : self.encoder(turn=turn+1)

    
    def decoder(self, turn:int):
        if turn > 1 :
            size_diff = (np.array(self.contract_path_mats[-1].shape) - np.array(self.mat.shape)) // 2
            cont_mat = self.contract_path_mats.pop()
            self.mat = torch.cat(
                (cont_mat[:,:,size_diff[2]:size_diff[2]+self.mat.shape[2],size_diff[3]:size_diff[3]+self.mat.shape[3]], self.mat), 
                dim=1) # modify image

        self.mat = self.module_list[turn+self.depth-1](self.mat)  
        self.mat = self.samplings[turn+self.depth-1](self.mat)

        if turn < self.depth+1 : self.decoder(turn=turn+1)


    def basic_cycle(self, in_ch:int, out_ch:int, k_size=3, stride_=1, padding_=0, bias_=True):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k_size,
                stride=stride_,
                padding=padding_,
                bias=bias_
                ),
            # nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
            )

        
    def init_network(self):
        enc_channels = [(self.image_ch, 64) if not i else ((2**(i-1)) * 64, (2**i) * 64) for i in range(self.depth)]
        dec_channels = [self.dec_ch // (2**i) for i in range(self.depth+1)]

        for in_, out_ in enc_channels:
            temp_module = []
            temp_module.append(self.basic_cycle(in_ch=in_, out_ch=out_))
            temp_module.append(self.basic_cycle(in_ch=out_, out_ch=out_))

            self.module_list.append(nn.Sequential(*temp_module))
            self.samplings.append(nn.MaxPool2d(kernel_size=2))

        for idx, ch_ in enumerate(dec_channels):
            temp_module = []
            temp_module.append(self.basic_cycle(in_ch=ch_ if idx else ch_//4, out_ch=ch_//2))
            temp_module.append(self.basic_cycle(in_ch=ch_//2, out_ch=ch_//4 if idx != self.depth else ch_//2))

            self.module_list.append(nn.Sequential(*temp_module))

            if idx != self.depth:
                self.samplings.append(nn.ConvTranspose2d(
                    in_channels=ch_//4, 
                    out_channels=ch_//4, 
                    kernel_size=2,
                    stride=2))
            else:
                self.samplings.append(nn.Conv2d(
                    in_channels=ch_//2,
                    out_channels=self.class_num if self.target_ch is None else 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

    
    def forward(self, mat_:torch.tensor):
        self.mat = mat_
        self.encoder(1)
        self.decoder(1)

        return self.mat


def data_loader(path_:str):
    transform = transforms.ToTensor()
    # need modify
    return transform(cv2.Mat(np.load(path_))).unsqueeze(0)