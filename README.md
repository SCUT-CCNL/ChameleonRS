# Adaptive Disentangled Target Representation for Unsupervised Domain Adaptation in Remote Sensing Segmentation

This is the official code of ChameleonRS. 
Paper web page: [Adaptive Disentangled Target Representation for Unsupervised Domain Adaptation in Remote Sensing Segmentation](https://authors.elsevier.com/a/1l7LJ3OWJ9CTXq).

## Citation
```

@article{LU2025111029,
title = {Adaptive disentangled target representation for unsupervised domain adaptation in remote sensing segmentation},
journal = {Engineering Applications of Artificial Intelligence},
volume = {156},
pages = {111029},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.111029},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625010292},
author = {Runuo Lu and Shoubin Dong and Jianxin Jia and Xusheng Wang and Kai Liu and Jinsong Chen and Shanxin Guo and Xiaorou Zheng},
keywords = {Unsupervised domain adaption, Target representation enhancement, Semantic segmentation, Multi-level feature alignment, Domain separation networks, Cross-domain remote sensing image},
}
```


## Data Preparation
Please visit the following URL to download the Vaihingen.zip and Postdam.zip files, and place them in the `data` folder: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx
or
   ```
   cd ChameleonRS/data
   wget https://seafile.projekt.uni-hannover.de/seafhttp/files/689f04af-4818-4e88-848d-78d09f090bcf/Potsdam.zip
   wget https://seafile.projekt.uni-hannover.de/seafhttp/files/59cd63e9-7099-4627-9a05-2777bbd37115/Vaihingen.zip
   ```

### Potsdam Dataset

Move the following files to the `dataset/Potsdam` folder:
- `3_Ortho_IRRG.zip`
- `2_Ortho_RGB.zip`
- `5_Labels_all_noBoundary.zip`
You can use these following commands to complete the operation:

1. Unzip the datasets:
   ```
   cd data
   unzip Potsdam.zip "Potsdam/3_Ortho_IRRG.zip" "Potsdam/2_Ortho_RGB.zip" "Potsdam/5_Labels_all.zip" -d dataset
   ```

2. Create the necessary directories:
   ```
   mkdir -p dataset/Potsdam/{IRRG,RGB,Label}
   ```

3. Unzip the respective files:
   ```
   unzip -j dataset/Potsdam/3_Ortho_IRRG.zip "3_Ortho_IRRG/*" -d dataset/Potsdam/IRRG/
   unzip -j dataset/Potsdam/2_Ortho_RGB.zip "2_Ortho_RGB/*" -d dataset/Potsdam/RGB/
   unzip -j dataset/Potsdam/5_Labels_all.zip "5_Labels_all/*" -d dataset/Potsdam/Label/
   ```

### Vaihingen Dataset

Move the following files to the `dataset/Vaihingen` folder:
- `ISPRS_semantic_labeling_Vaihingen.zip`
- `ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip`
You can use these following commands to complete the operation:

1. Unzip the datasets:
   ```
   cd data
   unzip Vaihingen.zip "Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip" "Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip" -d ./dataset
   ```

2. Unzip the respective files:
   ```
   unzip -j dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip "top/*" -d dataset/Vaihingen/images/
   unzip dataset/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip -d dataset/Vaihingen/labels
   ```

### Data Preprocessing

For **Potsdam Dataset**:
   ```

   python potsdamdata.py --source ./dataset/Potsdam --target ./Potsdam/RGB --split_size 512 --image_type RGB
   python potsdamdata.py --source ./dataset/Potsdam --target ./Potsdam/IRRG --split_size 512 --image_type IRRG
   ```

For **Vaihingen Dataset**:
   ```
   # Training data
   python vaihingendata.py --source ./dataset/Vaihingen --target ./Vaihingen --mode train
   # Validation data
   python vaihingendata.py --source ./dataset/Vaihingen --target ./Vaihingen --mode val
   # Test data
   python vaihingendata.py --source ./dataset/Vaihingen --target ./Vaihingen --mode test

   python read_data.py
   ```

---

## Installation

1. Create a new Conda environment:
   ```
   conda create -n ChameleonRS python=3.7
   ```

2. Activate the environment:
   ```
   conda activate ChameleonRS
   ```

3. Install required dependencies:
   ```
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
   ```

4. Install additional required libraries:
   ```
   pip install -r requirements.txt
   ```

---

## Training

### Potsdam IRRG to Vaihingen IRRG:

   ```
   bash dist_train.sh configs/dsn_pre.py 4 --work-dir my_workdir/IRRG_pre \
   && bash dist_train.sh configs/dsn_fpn_res50_512x512_20k.py 4 \
   --work-dir my_workdir/IRRG --load-from my_workdir/IRRG_pre/latest.pth
   ```

### Potsdam RGB to Vaihingen IRRG:
#### 1) Open the Configuration File

   Locate and edit the following file:
   ```
   configs/_base_/datasets/PVAda.py
   ```

#### 2) Find the Target Paths

   In the file, find the following three lines (around line XX-XX):
   ```
   # Current IRRG paths ¡ý
   img_dir='Potsdam/IRRG/img_dir/train',
   ann_dir='Potsdam/IRRG/ann_dir/train',
   split='Potsdam/IRRG/train.txt',
   ```

#### 3) Replace with RGB Paths

   Modify them to the following:
   ```
   # Modified RGB paths ¡ý
   img_dir='Potsdam/RGB/img_dir/train',
   ann_dir='Potsdam/RGB/ann_dir/train',
   split='Potsdam/RGB/train.txt',
   ```

#### 4) Train the Model

   Run the following command to train:
   ```
   bash dist_train.sh configs/dsn_pre.py 4 --work-dir my_workdir/RGB_pre \
   && bash dist_train.sh configs/dsn_fpn_res50_512x512_20k.py 4 \
   --work-dir my_workdir/RGB --load-from my_workdir/RGB_pre/latest.pth
   ```


---

## Testing

### 1. Potsdam IRRG to Vaihingen IRRG:

   ```
   #multi-gpu testing
   bash dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval ${EVAL_METRICS}
   #Examples
   bash dist_test.sh configs/dsn_fpn_res50_512x512_20k.py my_workdir/IRRG/iter_3500.pth 4 --eval mIoU

   #Test model and save the painted images for latter visualization.
   python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval ${EVAL_METRICS} --show-dir ${OUTPUT_FILE}
   #Examples
   python test.py configs/dsn_fpn_res50_512x512_20k.py my_workdir/IRRG/iter_3500.pth --eval mIoU --show-dir output/IRRG
   ```

### 2. Potsdam RGB to Vaihingen IRRG:

   ```
   bash dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval ${EVAL_METRICS}
   #Examples
   bash dist_test.sh configs/dsn_fpn_res50_512x512_20k.py my_workdir/RGB/iter_3500.pth 4 --eval mIoU

   #Test model and save the painted images for latter visualization.
   python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval ${EVAL_METRICS} --show-dir ${OUTPUT_FILE}
   #Examples
   python test.py configs/dsn_fpn_res50_512x512_20k.py my_workdir/RGB/iter_3500.pth --eval mIoU --show-dir output/RGB
   ```

---
## Acknowledgment
Our code is based on the method of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks for their work.