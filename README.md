# Tri-plane Text Fusion Network for Segmentation of Kidney Calculi in Veterinary CT Images (TTF-Net)
This repository provides the implementation of TTF-Net and the JBNU-VetCT dataset to reproduce the segmentation results of kidney, kidney stone, and renal pelvis from veterinary CT images.

## Update

### 🚀 Work in Progress


*Overview of the TTF-Net architecture.*  
![Fig 1 v14](https://github.com/user-attachments/assets/186c49aa-44b6-4028-9b21-8c4bdb6e77f1)  

*Detailed architecture of the proposed TTF module.*  
![Fig 2 v9](https://github.com/user-attachments/assets/3110449c-5544-41d0-a4b6-29758df38a82)  

## JBNU-VetCT Dataset
![Fig 3 v9](https://github.com/user-attachments/assets/8ad0739b-a627-4d4b-9642-62f1451c7986)
The data used in this study are available from the corresponding author upon reasonable request.  
(Corresponding author: Prof. Sang Jun Lee, sj.lee@jbnu.ac.kr)

## Usage:
### Recommended environment:
**Please run the following commands.**
```
conda create -n ttf-net python=3.8
conda activate ttf-net
```

### Training:
```
cd into ttf-net
python main_train.py --root <DATASET_ROOT_PATH> --dataset vetct --batch_size <BATCH_SIZE> --crop_sample <CROP_SAMPLE_NUM> --lr <LEARNING_RATE> --optim AdamW --max_iter <MAX_ITERATIONS> --eval_step <EVAL_INTERVAL> --cache_rate <CACHE_RATE> --num_workers <NUM_WORKERS> --output <OUTPUT_DIR> --contrastive_w <CONTRASTIVE_WEIGHT> --contrastive_t <CONTRASTIVE_TEMPERATURE> --gpu <GPU_ID>
```

## Acknowledgement
This repository makes extensive use of code from the [monai](https://monai.io/), [3D UX-Net](https://github.com/MASILab/3DUX-Net), [SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR), [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), and [EffiDec3D](https://github.com/SLDGroup/EffiDec3D), which have provided the basis for our framework.  
This project is licensed under the BSD 3-Clause License.  
We thank the authors for open sourcing their implementation.

## Citations

``` 
@inproceedings{
}
```
