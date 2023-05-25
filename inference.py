#!/usr/bin/env python3

from net.Unet import Unet
from net.UnetData import UnetData
from utils.save_load import *
from utils.IOU import *
from utils.read_arg import *

import os, cv2, json, time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms



def save_log(infer_path, epoch, loss, IOU, F_time, E_time):
    log_path = os.path.join(infer_path, "log")
    log_file_name = "log_%04d.txt"%epoch
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, log_file_name), "w", encoding="utf-8") as f:
        f.write(f"loss : {loss:.5f} %\n")
        f.write(f"IOU : {IOU*100:.5f} %\n")
        f.write(f"FPS : {F_time:.5f} sec\n")
        f.write(f"Inference time : {np.mean(E_time):.5f} sec\n")

    print("\n **  saved log  **")
    print(f" - loss           : {loss:.5f}")
    print(f" - IOU            : {IOU*100:.5f}")
    print(f" - FPS            : {F_time:.5f} sec")
    print(f" - Inference time : {np.mean(E_time):.5f} sec")



def save_img(infer_path, epoch, idx, output, total):
    save_path = os.path.join(infer_path, "img")
    file_name = "epoch_%04d_%02d.png" % (epoch, idx)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img = np.zeros((output.shape[2], output.shape[3], 3), dtype=np.uint8)
    img[(output[0,0,:,:] >= 1.0).cpu()] = np.array([255,255,255])
    
    print(f"save result ----- {idx+1} / {total}")
    cv2.imwrite(os.path.join(save_path, file_name), img)



def infer(args, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Available Device = {device}")
    cudnn.enabled = True
    
    # load argument -----------------------------------------------------
    file_name_ = args.pth
    pth_path_ = cfg["pth_path"]
    data_path_ = cfg["data_path"]
    infer_path_ = cfg["infer_path"]
    prefix_name = cfg["prefix_name"]
    batch_size_ = cfg["batch_size"]
    num_workers_ = cfg["num_workers"]
    depth_ = cfg["depth"]
    img_channel_ = cfg["img_channel"]
    target_channel_ = cfg["target_channel"]
    
    # dataset load ------------------------------------------------------
    print(f"Data init  " + "="*60)
    infer_data = UnetData(data_path_, mode="I", depth_=depth_, target_ch=target_channel_)
    infer_loader = DataLoader(infer_data, batch_size=batch_size_, num_workers=num_workers_, shuffle=False)
    class_num = len(infer_data.class_keys) if target_channel_ is None else 1
    print(f"Data init complete  " + "="*51)

    # create network ----------------------------------------------------
    model = Unet(class_num_=class_num, depth_=depth_, image_ch_=img_channel_, target_ch_=target_channel_).to(device)
    loss_func = DiceLoss_BIN(class_num, device).to(device)

    # initialize model --------------------------------------------------
    model, epoch = load_net(pth_path_, file_name_, prefix_name, model)

    with torch.no_grad():
        model.eval()
        loss_arr = []
        IOU_arr = []
        elapsed_time = []
        cnt = 0

        fps_start = time.time()
        for idx, i in enumerate(infer_loader):
            cnt += 1
            infer_input = i[0].to(device)
            infer_label = i[1].to(device)

            infer_start = time.time()
            infer_output = model(infer_input)
            infer_end = time.time()
            elapsed_time.append(infer_end - infer_start)

            infer_loss, IOU = loss_func(infer_output, infer_label)
            loss_arr += [infer_loss.item()]
            IOU_arr += [IOU.item()]

            save_img(infer_path_, 
                     epoch, 
                     idx, 
                     infer_output,
                     len(infer_data))
            
        fps_end = time.time()
        fps_time = (fps_end - fps_start) / cnt

        save_log(infer_path_, 
                 epoch, 
                 np.mean(loss_arr),
                 np.mean(IOU_arr),
                 fps_time,
                 elapsed_time)
        


if __name__ == "__main__":

    args = read_infer_arg()
    with open("./Unet_config.json") as f:
        cfg = json.load(f)

    infer(args=args, cfg=cfg)
