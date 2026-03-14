# RFMiD Retinal Disease Classification using CNN

This project implements deep learning models for **retinal disease classification** using the **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**.

The objective of this project is to compare different CNN architectures and analyze the impact of **Focal Loss** for handling class imbalance in medical image classification tasks.

---

## Dataset

Dataset used: **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**

Number of images: ~3200 retinal fundus images
Number of classes: 28 retinal disease categories

Each sample consists of a retinal fundus image and its corresponding disease label.

Dataset structure used for training:

```
Retinal-disease-classification/
├── labels.csv
└── images/
```

---

## Models Used

The following CNN architectures were evaluated using **transfer learning with pretrained ImageNet weights**:

* **ResNet18**
* **ResNet18 + Focal Loss**
* **ResNet50**
* **VGG16**

---

## Training Configuration

| Parameter     | Value     |
| ------------- | --------- |
| Batch Size    | 32        |
| Learning Rate | 1e-4      |
| Epochs        | 10        |
| Image Size    | 224 × 224 |
| Optimizer     | Adam      |

Framework: **PyTorch**
Hardware: **Tesla V100 GPU (TWCC)**

---

## Experiments

| Model    | Loss Function    |
| -------- | ---------------- |
| ResNet18 | CrossEntropyLoss |
| ResNet18 | FocalLoss        |
| ResNet50 | CrossEntropyLoss |
| VGG16    | CrossEntropyLoss |

These experiments allow comparison of:

* CNN architecture performance
* Training convergence behavior
* Effect of Focal Loss on model training

---

## Training Monitoring

All experiments were logged using **Weights & Biases (wandb)** for monitoring training metrics such as loss and validation accuracy.

The public project link is provided in:

```
wandb_log_link.txt
```

---

## Repository Structure

```
train_cnn.py        Main training script
rfmid_dataset.py    Dataset loader
focal_loss.py       Implementation of Focal Loss
wandb_log_link.txt  wandb experiment project link
result_report.pdf   Experiment results and analysis
```

---

## Notes

* Large files such as **dataset images** and **trained model weights** are excluded from this repository.
* Training was conducted on the **TWCC GPU environment using a Tesla V100 GPU**.
* Experiment results and detailed analysis are provided in **result_report.pdf**.
