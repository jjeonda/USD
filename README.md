# AAAI-2023

This contains the codes for our paper 889: Uncertainty-based One-phase Learning to Enhance Pseudo-Label Reliability for Semi-supervised Object Detection

## Requirements
code test setup: Ubuntu 16.04, NVIDIA Titan Xp with CUDA 9.0 and cuDNNv7, OpenCV 3.3.0

## Setup
YOLOv3 website instructions (https://pjreddie.com/darknet/yolo/)

## Dataset 
 - We tested our algorithm using PASCAL VOC and MS COCO dataset.
 - Pascal VOC
      - data root : /path/to/VOCdevkit/
      - pseudo-label root : /path/to/VOCdevkit/VOC_PL/
 - MS COCO 
      - data root : /path/to/coco/
      - pseudo-label root : /path/to/coco_PL/

 - For SSOD (Semi-supervised learnig for object detection), we set 
   - Pascal VOC
    ' VOC2007 trainset as the labeled dataset
    ' VOC2012 trainset as the unlabeled dataset
    ' VOC2007 testset for evaluation
   - MS COCO
    ' coco2014 validset (co-35k) as the labeled dataset
    ' coco2014 trainset (co-80k) as the unlabeled dataset
    ' coco2014 mini-val for evaluation

 -  list is included in our code
    ' trainval_VOC2007.txt (L) + trainval_VOC2012.txt (Un)  ===> FOR SSOD : trainval_VOC0712.txt
    ' test_VOC2007.txt


## Training 
Format of dataset : https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/

**Prepare FS (VOC07 only) model weight**
- backup/g-yolov3-tiny-voc07.weights
  - trained with labeled VOC07 trainset  

**Compile with proposed method**
Option : include/darknet.h
  - FN solution (uc weighted loss)
     - #define FN
     - Weighted by uncertainty for all negative sample
  - FP solution (adaptive filtering)
     - #define FP
  - Pseudo-label update 
     - #define PLU_auto
     
     ```Shell
     make
     ```

**Make pseudo-label**
```
# make pseudo-abel as inference results
./darknet detector valid cfg/voc_pl.data cfg/g-yolov3-tiny-voc_val.cfg ./backup/g-yolov3-tiny-voc07.weights
# move pseudo-label to pseudo-label root
mv results/pseudo_label /path/to/VOC_PL/labels
```
     
**Remove zero labeled data on the list**
```     
python scripts/rm_list.py 
# input: trainval_VOC0712.txt  # output: trainval_VOC0712_rm.txt
```
     
**Training**
```Shell     
./darknet detector train cfg/voc.data cfg/g-yolov3-tiny-voc.cfg darknet19_448.conv.23 
# if use multiple GPUs
./darknet detector train cfg/voc.data cfg/g-yolov3-tiny-voc.cfg darknet19_448.conv.23 -gpus 0,1,2,3
```
     
**(OPTION) Set PLU_auto**
Pseudo-label update at the point displayed on the screen 
- e.g.) find pseudo label update weight point (local min) : 500000!! --> point: 500000 weight
```
./darknet detector valid cfg/voc_pl.data cfg/g-yolov3-tiny-voc_val.cfg ./backup/g-yolov3-tiny-voc_(point).weights
# Move the updated pseudo-label to data root
mv results/pseudo_label /path/to/VOC_PL/labels
# Restart training from the updated point
./darknet detector train cfg/voc.data cfg/g-yolov3-tiny-voc.cfg ./backup/g-yolov3-tiny-voc_(point).weights
```
     
## Evaluation 
```
 ./darknet detector valid cfg/voc.data cfg/g-yolov3-tiny-voc_val.cfg /path/to/weights
```
**Ensemble**
Weighted-Boxes-Fusion: implemented based on https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- Ensemble Models(3)
  - FS 
  - update_point
  - trained with updated pseudo-label
  
- Make Model results file
```Shell
./darknet detector valid cfg/voc_wbf.data cfg/g-yolov3-tiny-voc_val.cfg /path/to/weights   --> results/for_wbf_results.json
```

- Get WBF evaluation
```Shell
cd scripts/Weighted-Boxes-Fusion/
python wbf_usd.py [input1] [input2] [input3]
```

## Result
We report the ablation study.

|    FN    |  FP(filtering+update)  |    Ensemble      |    mAP(%)     |
|:--------:|:----------------------:|:----------------:|:-------------:|
|          |                        |                  |     33.19     |
|     √    |                        |                  |     34.53     |
|     √    |           √            |                  |     36.53     |
|     √    |           √            |        √         |     37.29     |


