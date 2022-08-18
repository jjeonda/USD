# USD
# AAAI-2023

This is the PyTorch-SSD implementation of the paper "Uncertainty-based One-phase Learning to Enhance Pseudo-Label Reliability for Semi-supervised Object Detection".

## Enviornment details
Ubuntu 18.04.5    
CUDA 9.2   
Python version 3.7    
Pytorch version 1.2.0   
torchvision 0.4.0    

## Dataset
Using VOC2007 as labeled dataset and VOC2012 as unlabeled dataset.  

## Training step
**Training**
```
CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py  

# You can search for pseudo-label update point in train_ssd_gsm_ucfilter.py with keyword [update]
CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py --adaptive_filtering=True
```

**Training with proposed method**
  - FN solution (uc weighted loss)
     - Search key word : scale
     - Weighted by uncertainty for all negative sample
  - FP solution (adaptive filtering)
     - Search key word : adaptive
     - If you want to compare static filtering, find the static keyword and uncommend it.
  - Pseudo label update 
     - Search key word : update
     - (1) Find keyword and uncommend codes -> save update point file 
     - (2) Pseudo label update with saved file
     ```Shell
     # The save folder location is set in the file
     python wl_voc_gsm.py --trained_model=[save update point file]
     ```
     - (3) Resume with update file
     ```Shell     
     # Before train, update the pseudo label file(Annotations, txt list file) created in step (2) to voc0712.py
     CUDA_VISIBLE_DEVICES=[] python train_ssd_gsm_ucfilter.py --resume=[save update point file]
     ```
     
## Evaluation step
**Eval mAP(%)**
```
python eval_voc_gsm.py --trained_model=weights/ssd_300_120000.pth
```
**Ensemble**
(1) Make json file for ensemble 
```Shell
# File trained only with voc2007
python eval_voc_gsm.py --trained_model=[only07.pth]
# File before update
python eval_voc_gsm.py --trained_model=[save update point file]
# File after update
python eval_voc_gsm.py --trained_model=ssd_300_120000.pth
```
(2) Weight box Fusion
```Shell
cd Weighted-Boxes-Fusion/
# Input the three files in (1) -> make results
python wbf_3_voc.py [input1] [input2] [input3]
```
(3) Eval output results file
```Shell
cd scripts/
python reval_voc_py3.py --voc_dir=../../data/VOCDevkit/ --year=2007 --image_set=test --classes=voc.names [output_dir]
```

## Result
We report the ablation study.

|    FN    |  FP(filtering+update)  |    Ensemble      |    mAP(%)     |
|:--------:|:----------------------:|:----------------:|:--------------:|
|          |                        |                  |      71.8      |
|     √    |                        |                  |      72.3      |
|     √    |           √            |                  |      73.4      |
|     √    |           √            |        √         |      75.1      |


