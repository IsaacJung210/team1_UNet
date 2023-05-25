#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, json, cv2
import numpy as np
from glob import glob
from utils.ImgAug import *

from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

class UnetData(Dataset):
    def __init__(self, data_path, mode=None, depth_=4, target_ch=None):
        super(UnetData,self).__init__()
        self.target_ch = target_ch

        # load class information -------------------------------------------
        with open(os.path.join(data_path, "classes.json")) as f:
            self.classes = json.load(f)["class_num"]
            self.class_keys = list(self.classes.keys())

        # load files -------------------------------------------------------
        input_dir = os.path.join(data_path, "rgb")
        label_dir = os.path.join(data_path, "seg")

        if mode == "T": mode_dir = "train"
        elif mode == "I": mode_dir = "test"
        elif mode == "V": mode_dir = "val"
        else: raise Exception("Unknown Error")
        input_dir = os.path.join(input_dir, mode_dir)
        label_dir = os.path.join(label_dir, mode_dir)

        if not os.path.exists(input_dir) : raise Exception(input_dir + " folder not exist")        
        if not os.path.exists(label_dir) : raise Exception(label_dir + " folder not exist")    

        input_list = sorted(glob(os.path.join(input_dir, "*.png")))
        label_list = sorted(glob(os.path.join(label_dir, "*.png")))
        
        self.total_ = len(input_list)
        self.img_aug = ImgAug()

        self.input_ = []
        self.label_ = []

        cnt = 0
        for in_, la_ in zip(input_list, label_list):
            cnt += 1
            if in_.lstrip(input_dir) != la_.lstrip(label_dir) : raise Exception("files not same")
            print(f"<loading> {in_} ............ {cnt} / {self.total_}")
            self.input_.append(cv2.imread(in_))
            self.label_.append(cv2.imread(la_))
            # if cnt >= 10: break

        # calc input border size
        self.pad = 4 * (2**depth_ + sum([2**(d+1) for d in range(depth_)])) // 2


    def input_resize(self, index):
        resized_img = cv2.copyMakeBorder(self.input_[index], self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT101)
        b, g, r = cv2.split(resized_img)
        return np.array([b, g, r], dtype=np.float32)
    

    def normalization(self, i_, mean=0.5, std=0.5):
        return (i_ - mean)/std
    

    def label_init(self, index):
        ch_ = []
        if self.target_ch is None:            
            for class_ in self.class_keys:
                matches = np.all(self.label_[index] == self.classes[class_], axis=2).astype(np.float32)
                ch_.append(matches)
        else :
            matches = np.all(self.label_[index] == self.classes[self.class_keys[self.target_ch]], axis=2).astype(np.float32)
            ch_.append(matches)

        return np.array(ch_)


    def __getitem__(self, index):
        if index >= len(self.input_): raise IndexError
        i = self.input_resize(index=index)
        l = self.label_init(index=index)

        np.random.seed(seed=index)
        pick = np.random.randint(low=0, high=3)

        i, l = self.img_aug.aug_list[pick]((i, l))
        i = i / 255
        i = self.normalization(i)
        return [i, l]

    
    def __len__(self):
        return len(self.input_)
    



if __name__ == "__main__":
    test = UnetData("./Carla_Data01")
    eval_loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
    input_, label_ = test[1]

    plt.subplot(122)
    plt.hist(label_.flatten(), bins=20)
    plt.title('label')

    plt.subplot(121)
    plt.hist(input_.flatten(), bins=20)
    plt.title('input')

    plt.tight_layout()
    plt.show()