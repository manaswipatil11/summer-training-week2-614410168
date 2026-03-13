# RFMiD Retina Disease Classification using CNN

This project implements CNN-based retinal disease classification using the **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**. The objective is to compare different deep learning architectures and analyze the impact of different loss functions.

## Dataset

Dataset used: **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**
Number of images: ~3200 fundus images
Number of classes: 28 retinal disease categories

Each sample consists of a retinal image and its corresponding disease label.

Dataset structure used for training:

```
Retinal-disease-classification/
├── labels.csv
└── images/
```

## Models Used

The following CNN architectures were evaluated:

* **ResNet18**
* **ResNet18 with Focal Loss**
* **ResNet50**
* **VGG16**

## Training Configuration

| Parameter     | Value   |
| ------------- | ------- |
| Batch Size    | 32      |
| Learning Rate | 1e-4    |
| Epochs        | 10      |
| Image Size    | 224x224 |
| Optimizer     | Adam    |

## Experiments

| Model    | Loss Function    |
| -------- | ---------------- |
| ResNet18 | CrossEntropyLoss |
| ResNet18 | FocalLoss        |
| ResNet50 | CrossEntropyLoss |
| VGG16    | CrossEntropyLoss |

## Training Monitoring

All training runs were logged using **Weights & Biases (wandb)**.

wandb project link:
(see `wandb_log_link.txt`)

## Files in this Repository

```
train_cnn.py        - Main training script
rfmid_dataset.py    - Dataset loader
focal_loss.py       - Implementation of Focal Loss
wandb_log_link.txt  - wandb experiment link
result_report.pdf   - Experiment results and analysis
```

## Notes

* Large files such as dataset images and trained model weights are excluded from this repository.
* Training was performed on **TWCC GPU environment using Tesla V100**.
