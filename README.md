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
