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
git clone https://github.com/rohit18247/pgimer_assignment/
cd <move in to the directory>
```

### 2. Create and Activate the Python Environment (it can be a new or a pre-existing one)

You can use `conda` or `virtualenv`.

**Using conda:**
```bash
conda create -n pneumonia_env python=3.8
conda activate pneumonia_env
```

### 3. Install Dependencies (installing the packages - most important)

```bash
pip install -r requirements.txt
```

Or if you prefer using `conda`:
```bash
conda env create -f environment.yml
conda activate pneumonia_env
```

### 4. Check PyTorch MPS/CPU/GPU Support (essential)

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
```

---

## Dataset

Place the dataset `pneumoniamnist.npz` in the root folder of the project. Downloaded from the kaggle link as provided in the assignment

---

## Running the Notebook

Launch Jupyter Notebook (you can also use jupyter lab):
```bash
jupyter notebook
```
Open and run the notebook as directed within it.

---
## Notes

- It was trained and tested on the Mac M1 (with MPS support).
- Adapted for CPU/GPU usage if MPS unavailable.
