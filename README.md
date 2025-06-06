# MGFI-Net


Many convolutional neural networks have been
applied to medical image segmentation tasks and have shown
excellent model performance. However, existing network archi-
tectures still exhibit significant limitations in accurately iden-
tifying complex structures. To address these issues, a Global
Semantic Guidance (GSG) module is proposed in this study,
which enhances global semantic information in the encoder stage
and preserves spatial detail information. To further improve
the model’s adaptability to contextual variations, a Layer-Wise
Feature Integration Unit (LFIU) and the Decoder Semantic
Enhancement Module (DSEM) have been proposed. The former
leverages multi-level semantic information from the encoder
to provide multi-scale information at different levels to the
decoder. The latter uses wavelet transform and channel attention
mechanisms to improve the feature fusion scheme. In addition,
based on the integration of GSG, LFIU, DSEM, and the pre-
trained InceptionNext module, a MGFI-Net network is proposed.
To validate the effectiveness of the proposed model, two groups
of experiments have been set. First, the combination of GSG,
LFIU, and DSEM is integrated into four popular U-shaped
models. Second, the proposed MGFI-Net has been tested by using
four commonly used datasets (ISIC2018, BUSI, Kvasir-SEG, and
CVCClinicDB). Experimental results show that the integration
of GSG, LFIU, and DSEM can enhance model performance, and
the proposed MGFI-Net outperforms other popular models in
medical image segmentation tasks.

# Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:\
ISIC-2018 (dermoscopy, 2,594 images fortraining, 100 images for validation, and 1,000 images for testing)\
Kvasir-SEG (gastrointestinal polyp, 600 images for training, 200images for validation, and 200 images for testing)\
BUSI (breast ultrasound, 399 images for training.113 images for validation, and 118 images for testing)\
CVC-ClinicDB (colorectal cancer, 367 images for training, 123images for validation, and 122 images for testing)\
The dataset path may look like:
```
/The Dataset Path/
├── ISIC-2018/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```
 # Usage
 Installation
 ```
 git clone git@github.com:shen123shen/MGFI-Net.git
 conda create -n shen python=3.8
 conda activate shen
 conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Training
 ```
python train_cuda.py
 ```
Evaluation
 ```
python Test.py
 ```
