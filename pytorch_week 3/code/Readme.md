PyTorch Week 3: Deep Learning Project 🚀
This repository contains the complete solution for a deep learning project focusing on implementing and analyzing two core neural network architectures from scratch using PyTorch: a ResNet for image classification and a Transformer for sequence-to-sequence tasks.

Table of Contents
Project Overview

File Structure

Requirements

How to Run

Project Deliverables & Results

Reports

<br>

1. Project Overview
This project implements two fundamental deep learning models from their basic building blocks (nn.Conv2d, nn.Linear, etc.) without relying on high-level pre-built architectures from libraries like torchvision.models.

ResNet-18 for Image Classification: Trained on the CIFAR-10 dataset to classify images. The implementation includes:

A custom ResNet18 model built from scratch.

Training and evaluation loops.

Visualization of training curves (loss and accuracy).

A normalized confusion matrix.

Visualization of correct and incorrect predictions.

Grad-CAM heatmaps to visualize model attention.

Transformer for Sequence-to-Sequence (Machine Translation): A simplified Transformer model is implemented and trained on a toy translation dataset. Key components include:

Custom MultiHeadAttention, PositionalEncoding, EncoderLayer, and DecoderLayer.

Training and evaluation of the complete Transformer model.

Visualization of training loss.

Calculation of the BLEU score for translation quality.

Visualization of self-attention heatmaps to show how the model weighs different input words.

<br>

2. File Structure
The repository follows a clean, organized structure to separate code, data, and results. You can click on the file and folder names below to navigate directly to them on GitHub.

pytorch-week3/
├── [code/](code/)                          # Main directory for all source code
│   ├── [custom_resnet.py](code/custom_resnet.py)     # ResNet-18 architecture from scratch
│   ├── [custom_transformer.py](code/custom_transformer.py) # Transformer architecture from scratch
│   ├── [train_resnet.py](code/train_resnet.py)       # Script to train and evaluate ResNet
│   └── [train_mt.py](code/train_mt.py)           # Script to train and evaluate Transformer
├── [data/](data/)                          # (Automatically created) Stores the dataset
│   └── [cifar-10-batches-py/](data/cifar-10-batches-py/) # CIFAR-10 dataset
├── [runs/](runs/)                          # (Automatically created) Stores all output and results
│   ├── [cls/](runs/cls/)                     # Results for the classification task
│   │   ├── resnet18_cifar10.pth
│   │   ├── curves_cls.png
│   │   ├── confusion_matrix.png
│   │   ├── preds_grid.png
│   │   ├── miscls_grid.png
│   │   └── gradcam_*.png
│   └── [mt/](runs/mt/)                       # Results for the machine translation task
│       ├── transformer.pth
│       ├── curves_mt.png
│       ├── bleu_report.txt
│       ├── decodes_table.md
│       ├── attention_*.png
│       └── masks_*.png
└── [report/](report/)                      # Project reports
    ├── [report.md](report/report.md)                 # Detailed project report
    └── [one_page_visual_report.md](report/one_page_visual_report.md) # Visual report with key figures

<br>

3. Requirements
To run the code, you need to install the necessary Python libraries.

pip install torch torchvision matplotlib scikit-learn nltk numpy

Note: Make sure your PyTorch installation is compatible with your hardware (CPU or GPU).

<br>

4. How to Run
Follow these simple steps to execute the training scripts and generate the results.

Clone the repository:

git clone [https://github.com/your-username/pytorch-week3.git](https://github.com/your-username/pytorch-week3.git)
cd pytorch-week3

Run the ResNet classification script:

python code/train_resnet.py

This script will train the ResNet model on CIFAR-10, save the model weights, and generate all classification-related plots in the runs/cls directory.

Run the Transformer translation script:

python code/train_mt.py

This script will train the Transformer model on the toy dataset, save the model weights, and generate all translation-related artifacts in the runs/mt directory.

<br>

5. Project Deliverables & Results
Upon successful execution, the runs/ folder will be populated with the following output files, providing a comprehensive view of the models' performance and behavior.

Classification (runs/cls)

resnet18_cifar10.pth: The trained ResNet-18 model.

curves_cls.png: Shows training and validation loss/accuracy over epochs.

confusion_matrix.png: A heatmap of classification performance.

preds_grid.png & miscls_grid.png: Grids of correctly and incorrectly classified images.

gradcam_*.png: Visualizations showing which parts of an image the model "paid attention to" for its prediction.

Machine Translation (runs/mt)

transformer.pth: The trained Transformer model.

curves_mt.png: Shows the training loss for the Transformer.

bleu_report.txt: A file containing the final BLEU score.

decodes_table.md: A markdown table comparing source, ground truth, and decoded sentences.

attention_*.png: Heatmaps illustrating the self-attention mechanism within the encoder layers.

masks_*.png: Visualizations of the source padding and target causal masks.

<br>

6. Reports
The detailed project analysis and a one-page visual summary are available in the report/ directory. These reports explain the architectural choices, training process, and a detailed analysis of the generated results.