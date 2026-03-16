# Learning-to-Select-Demonstrations

## Setup

```bash
git clone https://github.com/Yuchi017/Learning-to-Select-Demonstrations.git
```
```bash
cd Learning-to-Select-Demonstrations
```
```bash
conda create -n LSD python=3.10 -y
```
```bash
conda activate LSD
```
```bash
pip install -r requirements.txt
```

## Process Data

### 1. Download Dataset

Download the datasets from the following sources:

- **Age Prediction (UTKFace)**  
  https://susanqq.github.io/UTKFace/

- **Aesthetic Score (AVA)**  
  https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment

- **Facial Beauty (SCUT-FBP5500 v2)**  
  https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores

- **Wild Image Quality (KonIQ-10k)**  
  https://database.mmsp-kn.de/koniq-10k-database.html

- **Modified Image Quality (KADID-10k)**  
  https://database.mmsp-kn.de/kadid-10k-database.html

### 2. Process Dataset
```bash
./script/process_data.sh
```

### 3. Update datasets.yaml

## LSD Training (DQN)
```bash
./script/train_dqn.sh
```