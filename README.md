# pgimer_assignment
Repository for PGIMER assignment, containing code notebook, requirements.txt, readme, along with a PPT as instructed.

# Instructions mentioned here for reproducibility: 

# Pneumonia Detection using ResNet-50

Author: **Rohit Gupta**  
Contact: rohit.gupta.delhi1995@gmail.com

## Overview

This project implements a binary image classification model using a pre-trained ResNet-50 to detect pneumonia from chest X-ray images (PneumoniaMNIST dataset). It includes strategies for handling class imbalance, preventing overfitting, and uses visualizations like t-SNE and UMAP to analyze learned features.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and Activate a Python Environment

You can use `conda` or `virtualenv`.

**Using conda:**
```bash
conda create -n pneumonia_env python=3.8
conda activate pneumonia_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you prefer using `conda`:
```bash
conda env create -f environment.yml
conda activate pneumonia_env
```

### 4. Check PyTorch MPS/CPU/GPU Support

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
```

---

## Dataset

Place the dataset `pneumoniamnist.npz` in the root folder of the project. You can download it from [MedMNIST](https://medmnist.com/).

---

## Running the Notebook

Launch Jupyter Notebook:
```bash
jupyter notebook
```
Open and run the notebook `pgimer_ass_rohit_final.ipynb` step-by-step.

---

## Key Features

- **Model Architecture:** Pre-trained ResNet-50 fine-tuned for binary classification
- **Loss Function:** CrossEntropy with class weighting
- **Optimizer:** Adam with weight decay for regularization
- **Augmentation:** RandomResizedCrop, Rotation, Horizontal Flip, Affine transforms
- **Class Imbalance:** Weighted sampling + class weights
- **Evaluation Metrics:** Accuracy, F1-Score, Confusion Matrix, ROC-AUC
- **Dimensionality Reduction:** t-SNE and UMAP for learned feature visualization
- **Early Stopping:** Implemented to avoid overfitting

---

## Reproduce the Training

```python
# in notebook:
train(model_res, train_loader, val_loader, criterion, optimizer, epochs=10)
```

---

## Results

- **Validation Accuracy:** ~96%
- **Test Accuracy:** ~82%
- **ROC-AUC:** Visualized for qualitative model assessment

---

## Visualizations

- Confusion Matrix
- ROC Curve
- t-SNE and UMAP plots for feature space analysis

---

## Notes

- Trained and tested on Mac M1 (with MPS support).
- Adapted for CPU/GPU usage if MPS unavailable.

---

## License

This project is licensed under MIT License.
