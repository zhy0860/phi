# SAR-NAS
Code accompanying the paper  
SAR-NAS: Skeleton-based Action Recognition via Neural Architecture Searching  
Accepted by the Journal of Visual Communication and Image Representation.  

## Requirements
```Python >= 3.5.5, PyTorch == 0.4.1, torchvision == 0.2.1.```  

## Datasets 
Skeleton data in NTU RGB+D can be obtained via:

https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H

Or via (Kinetics+NTU)：

https://pan.baidu.com/s/1O1azJwxkzh04cOuSWyXi1Q  
extracted code: data


## Architecture search
```python train_search_ntu.py```    

## Architecture evaluation
```python train_ntu.py```  

## Confusion matrix 
```python draw_confusion_matrix.py```  

## Visualization
Package graphviz is required to visualize the learned cells  
```python visualize.py DARTS```    
where DARTS can be replaced by any customized architectures in genotypes.py  

## Citation  
If you use any part of this code in your research, please cite our paper:  
@article{Zhang2020sar-darts,  
  title={SAR-NAS: Skeleton-based Action Recognition via Neural Architecture Searching},  
  author={Zhang, Haoyuan and Hou, Yonghong and Wang, Pichao and Guo, Zihui and Li, Wanqing},  
  journal={Journal of Visual Communication and Image Representation, preprint},  
  year={2020}  
}  

# DFN

------
## Requirements


----------

 - tensorflow-gpu (1.8.0)
 - python (3.6.10)
 - io, os, sys, math, random, numpy, scipy.misc, cv2
 - time, datatime, threading, cStringIO, ruamel_yaml, h5py
## Preparation (config.yaml)
(modality: There are 8 modalities here including ffd, ffr, tfd, tfr, tsd, tsr, ttd and ttr, detailed description is as follows.
modality 1: Primary modality
modality 2: Privilege modality)

----------
(Example: If the current modality in config.yaml is ffd and you would like to use ffr instead, take the first path as example, just modify "../label/datalist/ffd_test.txt" to "../label/datalist/ffr_test.txt".)

### train_test

 - rtesting_datalist: modality 1
 - stesting_datalist: modality 2
    -    ffd: FPV_first_depth
    -    ffr: FPV_first_RGB
    -    tfd: TPV_front_depth
    -    tfr: TPV_front_RGB
    -    tsd: TPV_side_depth
    -    tsr: TPV_side_RGB
    -    ttd: TPV_top_depth
    -    ttr: TPV_top_RGB
 - weight_path, seq_len, num_classes
 - dataset_name
    -    ffd_tsr: use FPV_first_depth(modality 1) and TPV_side_RGB(modality 2) for training
### train
 - rtraining_datalist: modality 1
 - straining_datalist: modality 2
### test
test_dataset: modality 2

## Running the code


----------
 The structure of project directories is the following:

        LUPI/
            data/
                First_RGB_Frame_224/
                ...
            LCK_DFN/
                code/
                    config.yaml
                    training_rgb_full.py
                    test4rgb_full.py
                    ...
                label/
                    label/
                    datalist/
                result/
                    model_weight/
                    npy/
            ADMD(PAMI)/
Confirm all options (config.yaml)
  

    python training_rgb_full.py
    python test4rgb_full.py


