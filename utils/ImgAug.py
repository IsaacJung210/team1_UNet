#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImgAug:
    def __init__(self):
        self.aug_list = [self.bright, self.contrast, self.flip_dim, self.rotation, self.shear]


    def bright(self, imgs:tuple, min_=10, max_=20):
        input_, label_ = imgs
        input_ = input_.astype(np.uint16)
        bright = int(np.random.uniform(min_, max_))
        input_ = np.clip(input_ + bright, 10, 255).astype(np.uint8)
        return (input_.astype(np.float32), label_)
    

    def contrast(self, imgs:tuple, min_=0.1, max_=0.3):
        input_, label_ = imgs
        input_ = input_.astype(np.uint16)
        alpha = np.random.uniform(min_, max_)
        input_ = np.clip(input_*(1+alpha) - 128*alpha, 0, 255).astype(np.uint8)
        return (input_.astype(np.float32), label_)


    def flip_dim(self, imgs:tuple, axis=1):
        input_, label_ = imgs
        input_ = np.array([cv2.flip(i, axis) for i in input_])
        label_ = np.array([cv2.flip(i, axis) for i in label_])
        return (input_, label_)


    def rotation(self, imgs:tuple, angle=5.0):
        input_, label_ = imgs
        angle = int(np.random.uniform(-angle, angle))

        in_h, in_w = input_.shape[1:]
        in_M = cv2.getRotationMatrix2D((int(in_w/2), int(in_h/2)), angle, 1)

        la_h, la_w = label_.shape[1:]
        la_M = cv2.getRotationMatrix2D((int(la_w/2), int(la_h/2)), angle, 1)

        input_ = np.array([cv2.warpAffine(i, in_M, (in_w, in_h), borderMode=cv2.BORDER_REFLECT) for i in input_])
        label_ = np.array([cv2.warpAffine(i, la_M, (la_w, la_h), borderMode=cv2.BORDER_REFLECT) for i in label_])
        label_ = (label_ > 0.5).astype(np.float32)
        
        return (input_, label_)
    

    def shear(self, imgs:tuple, r=0.5):
        input_, label_ = imgs
        r = np.random.uniform(-r, r)
        x = np.random.uniform(0.2, 0.5)

        if r <= 0 : x *= -1
        aff = np.array([[1, x, 0], [0, 1, 0]], dtype=np.float32)

        input_ = np.array([cv2.warpAffine(i, aff, input_.shape[1:], borderMode=cv2.BORDER_REFLECT) for i in input_])
        label_ = np.array([cv2.warpAffine(i, aff, label_.shape[1:], borderMode=cv2.BORDER_REFLECT) for i in label_])
        label_ = (label_ > 0.5).astype(np.float32)

        return (input_, label_)
