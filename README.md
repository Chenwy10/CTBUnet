# CTBUnet: A CNN-Transformer Combined Bilateral U-Net for Microscopy Image Enhancement of Live-Cell

## Netword architecture:
<p align="center"><img width="90%" src="BasicSR-github/data/CTBUnet.jpg" /></p>

## Dataset
The dataset is in Basicsr-github/datasets

## Installation
- basicsr.1.4.2
- pytorch.2.1.2

## Results and models
**On Synthesized WBC**

|Methods | Scale | PSNR | SSIM | mDice | mIoU |
|--------|-------|------|------|-------|------|
|CTBUnet |  x4   |31.20 |0.8979|0.9310 |0.9215|
|CTBUnet |  x8   |27.36 |0.8239|0.8940 |0.8731|

**On Real-world Sperm**

|Methods | Scale | PSNR | SSIM | mDice | mIoU |
|--------|-------|------|------|-------|------|
|CTBUnet |  x2.5 |25.88 |0.7855|0.6588 |0.6469|
|CTBUnet |  x5   |25.34 |0.7627|0.5909 |0.6076|

## Evaluation

### Pretrained models

The pretrained models are available in [google drive]()

### Evaluation process

```
python 
```