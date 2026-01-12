# CTBUnet: A CNN-Transformer Combined Bilateral U-Net for Microscopy Image Enhancement of Live-Cell

## Netword architecture:
<p align="center"><img width="90%" src="BasicSR-github/data/CTBUnet.jpg" /></p>

## Sample output:
<p align="center"><img width="90%" src="BasicSR-github/data/result1.jpg" /></p>

## Dataset
The datasets are in Basicsr-github/datasets

## Installation
- basicsr.1.4.2
- pytorch.2.1.2

## Results and models
**On Sperm Cell Enhancement Dataset**

|Methods | Scale | PSNR | SSIM | mDice | mIoU | Download |
|--------|-------|------|------|-------|------|-----------|
|CTBUnet |  x2.5 |25.88 |0.7855|0.6588 |0.6469|[GoogleDrive](https://drive.google.com/file/d/1Nvm6p3TQNoEiLT3E08372wd2zc5m_G6q/view?usp=sharing)|
|CTBUnet |  x5   |25.34 |0.7627|0.5909 |0.6076|[GoogleDrive](https://drive.google.com/file/d/1pEcS8ZRcgGAtEtP5skYXc_rVfTSbYdOo/view?usp=sharing)|

## Evaluation

### Evaluation process

```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/CTBUnet/test_CTBUgan_x2.5plus_Sperm_pairdata_no_half.yml
```

## Contact
Should you have any question, please contact chenwy.chen@mail.utoronto.ca


**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox
