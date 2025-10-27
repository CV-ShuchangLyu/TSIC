# The Teacher-Student Interactive Cycle: Joint Optimization with Inner-Loop Self-Distillation in Prompted Foundation Models for Efficient Semantic Segmentation
## This paper has been submitted to TCSVT.
This repo is the implementation of "The Teacher-Student Interactive Cycle: Joint Optimization with Inner-Loop Self-Distillation in Prompted Foundation Models for Efficient Semantic Segmentation". We refer to  [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Many thanks to SenseTime and their excellent repo.

<table>
    <tr>
    <td><img src="PaperFig\TSIC.png" width = "100%" alt="TSIC"/></td>
    </tr>
</table>

## Dataset Preparation

We select Cityscapes, PASCAL VOC (aug), CamVid and ADE20k as benchmark datasets. You can download these datasets from official website and put them under the "/data" folder.

## TSIC

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.11
        
    cuda >= 11.7

   **This version depends on mmengine and mmcv (2.0.1)**
    
3. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd TSIC
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```
### In reviewing stage, we provide the ''DV3-R18'' based framework for training and testing
### Training
##### cityscapes task:

     ```
     cd TSIC
     
     ./tools/dist_train.sh ./experiments/DeeplabV3/SAM_deeplabv3_r18b-d8_4xb2-80k_cityscapes-512x1024_KD.py 2
     ```

### Testing
##### cityscapes task:
Trained with the above commands, you can get your trained model to test the performance of your model.   

     ```
     cd TSIC
     
     /tools/dist_test.sh ./experiments/DeeplabV3/SAM_deeplabv3_r18b-d8_4xb2-80k_cityscapes-512x1024_KD.py ./experiments/DeeplabV3/SAM_DeeplabV3_KD_results/xxx.pth 2
     ```

If you have any question, please discuss with me by sending email to limenglm@buaa.edu.cn and lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
  title={Unsupervised Domain Adaptation for VHR Urban Scene Segmentation via Prompted Foundation Model-Based Hybrid Training Joint-Optimized Network}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2025.3564216}}
```
