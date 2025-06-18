# Chest X-Ray Classification System

img_class is a deep learning framework for chest X-ray image classification and Clinical Severity Index (CSI) scoring. The system supports multiple state-of-the-art models for medical image analysis and provides comprehensive evaluation metrics.

## Overview

This project implements a multi-output deep learning system that:
- Classifies chest X-ray images into 3 categories
- Generates CSI scores for 6 different lung regions  
- Calculates mean CSI scores for overall assessment
- Supports multiple model architectures for comparison

## Supported Models

- **Rad_Dino**: Microsoft's radiology-focused Vision Transformer (default)
- **CheXNet**: DenseNet-based architecture for chest X-ray analysis
- **ResNet34**: Classic residual network architecture
- **Vision Transformer (ViT)**: Standard transformer-based model

## Features

- Multi-output prediction (classification + regional CSI scoring)
- DICOM image preprocessing pipeline
- TensorBoard integration for training visualization
- Automatic model checkpointing based on validation performance
- Comprehensive evaluation metrics (precision, recall, F1-score, Cohen's kappa)
- Support for mixed precision training
- Configurable hyperparameters via command line

## Project Structure

img_class/ 
├── main.py # Main training script 
├── exp/ │ └── exp_main.py # Experiment management class 
├── models/ │ 
            ├── vit_rad_dino.py # Microsoft Rad-DINO model │ 
            ├── cheXNet.py # CheXNet implementation │
             ├── resnet.py # ResNet architecture │ 
             └── vision_trans.py # Vision Transformer 

├── data_provider/ │ └── dataloader.py # Data loading and preprocessing 
├── utils.py # Utility functions 
├── assess_result.py # Model evaluation script 
└── dicom_images_preprocess.py # DICOM preprocessing utilities



## Data Requirements

The system expects data organized as follows:
- `data_1/new_pil_images/`: Directory containing preprocessed chest X-ray images
- `data_1/labels.csv`: CSV file with image labels and CSI scores

### CSV Format
The labels.csv should contain columns for:
- `number`: Patient/image identifier
- Classification labels (3 classes)
- CSI region scores (6 regions)
- Mean CSI scores

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chest_rad/paradise

2 - Install dependencies:
pip install -r requirements.txt

3 - Prepare your data:
Place chest X-ray images in data_1/new_pil_images/
Ensure data_1/labels.csv contains proper labels and CSI scores

Usage
Basic Training
Run with default parameters (Rad_Dino model): python main.py

Custom Configuration    
    python main.py \
    --model Rad_Dino \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --data_dirr data_1


Available Arguments
    --device: Computing device (auto-detected: cuda/cpu)
    --model: Model architecture (Rad_Dino, cheXNet, ResNet, initial_vit)
    --batch_size: Training batch size (default: 8)
    --learning_rate: Optimizer learning rate (default: 1e-5)
    --num_epochs: Number of training epochs (default: 30)
    --data_dirr: Data directory path (default: data_1)
    --experiment_path: Path to save model weights (default: experiments/)
    --use_amp: Enable mixed precision training (default: False)

Monitoring Training
The system integrates with TensorBoard for real-time monitoring:
    tensorboard --logdir experiments/Rad_Dino/runs/