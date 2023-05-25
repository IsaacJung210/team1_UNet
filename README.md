# Improvement of semantic segmentation Model
We're optimizing our real-time lane detection by reducing the weight of the segmentation model.<br/><br/>

## project period 
- 2023.03.16 ~ 2023.03.31

<br/> 

## Team members
- Kim Junhyung (https://github.com/Jun-WFI-hyung)
- Jung JiHoon (https://github.com/IsaacJung210)
- Lee SeWoong (https://github.com/tomy807)
- Park Jin (https://github.com/parkjin0903)

<br/>

## Used
- Carla
- RTAB-Map
- ROS
- U-Net
- E-Net
- DeeplabV3
- Segformer
- CityScapes DataSet

<br/>

## Overview
<br/>

### 1. Benchmark

    Compare team members' segmentation models and select the best one for improvement.
     - U-Net
     - E-Net
     - Deeplabv3+
     - Segformer
<img src="data\img\Benchmark.png" width="750" height="150">


<br/>

#### 1-1) Data

    Cityscapes

<img src="data\img\Benchmark_data.png" width="750" height="250">


<br/>

#### 1-2) Performance Results 
<br/>

    Selected U-net after excluding Segformer, E-net, and Deplabv3+ due to issues with chipset, 
    low MIoU, and significantly low inference time, respectively.
<img src="data\img\Performance_Results.png" width="750" height="400">
<br/><br/>

### 2.  U-net performance improvement

<br/>

#### 2-1) Data
    Used carla because Data labeling is good and most importantly, it's easy to implement in real time

<img src="data\img\Dataset.png" width="750" height="220">
<br/><br/>

#### 2-2) Learning with Binary cross entry
<img src="data\img\Binary.png" width="750" height="300">

    - 2,200 randomly selected images per width out of 4,375 train images were used for training up to 10 widths.
    - Loss converged quickly within 22,000 chapters.
    - IOU results showed fluctuations as seen in the second graph.
<br/>

#### 2-3) Learning with Dice Loss
<img src="data\img\Dice.png" width="750" height="300">

    - Loss hovering without convergence
    - Convergence observed with Dice loss and BCE loss together
    - Same learning environment and conditions as BCE loss
    - Slower convergence compared to BCE loss
    - IOU stable and more reliable than BCE loss, with occasional unexpected zero values
<br/>

#### 2-4) Compare
<img src="data\img\Binary_and_Dice.png" width="750" height="300">
<br/><img src="data\img\Binary_and_Dice2.png" width="500" height="150">

    - Similar patterns observed in graphs between BCE Loss and BCE+Dice Loss training
    - BCE+Dice Loss showed slower convergence but faster improvement in IOU metric, leading 
      to overall improvement.
    - Test conducted on Chapter 495 images
    - Model learned with DiceLoss showed 10% IOU average increase
    - Issues with inaccurate figures due to zero IOU cases such as noise.
<br/>

### 3. Lightening & Quantization
#### 3-1) Lightening
    - Choose between efficient model design or parameter reduction for a lightweight model
    - Unet model has a simple enough structure to focus on parameter reduction.
<br/>

#### 3-2) Quantization
<img src="data\img\Quantization.png" width="750" height="250">

    - Left graph shows trade-off between accuracy and latency
    - Right graph shows 4-bit quantized model has better accuracy and smaller size compared 
      to 8-bit quantized model
    - Better quantization with a larger network can be better in terms of performance and model size 
      compared to rough quantization with a smaller network
<br/>
<img src="data\img\Quantization.png" width="750" height="250">

    - Applied static and quantization awareness training to models
    - Used static quantization in TensorRT and PTQ (post-training quantization)
    - Attempted dynamic quantization, but not suitable for convolutional operations.
<br/>

#### 3-3) QAT (Quantization-Aware Training)
<img src="data\img\QAT.png" width="750" height="250">
    
    - Quantization parameters can be calculated based on collected statistics or learned during training
    - Float values are rounded to mimic int8 values, but calculations are still performed as floating 
      point numbers
    - This method typically provides higher accuracy after quantization
<br/>

#### 3-4) Applied quantization model structure
<img src="data\img\Applied quantization model structure.png" width="750" height="330">

    - Create pth files from enhanced performance Unet models
    - Quantify using PTQ, PTQ model and QAT
    - Use minmaxobserver to store output values as unsigned int8 and weight values as signed int8
    - Find optimal values by collecting Tensor statistics (e.g. min and max values) and fine-tuning
<br/>

#### 3-5) Comparison of quantization / dequantization test results
<img src="data\img\Results1.png" width="750" height="330">

    - Most time spent on convolution and inverse convolution
    - Changed to quantized layer, reduced convolution time from 67ms to 27ms
    - Inference speed improved by 50% or more overall, taking 1.6 seconds and 720 ms to deduce a photo.
<br/>

<img src="data\img\Results2.png" width="750" height="330">

    - Storing log based on 100 chapters
    - Reduced loss values but introduced noise in inferred image
    - QAT conducted from 10epoch to 1000epoch
    - Loss and IOU values tend to get worse as epoch increases
    - Model size reduced from 280 mb to 93 mb
<br/>

### 3-6) Conclusion

<img src="data\img\Conclusion1.png" width="750" height="300">
<img src="data\img\Conclusion2.png" width="750" height="300">

<br/><br/>

---
# How to use
- Check and customize Unet_config.json
- train : python train.py [T or F : load pth True or False]
  - optional arg: -p unet_epoch000.pth [Put in pth-filename]
  - e.g. python train.py T -p unet_epoch075.pth

 - test : python inference.py [Put in pth-filename]
   - e.g. python inference.py unet_epoch075.pth

