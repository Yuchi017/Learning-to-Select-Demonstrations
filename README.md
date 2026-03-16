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

### 1. Download Dataset from Website

#### links
    - Age Prediction(UTK): https://susanqq.github.io/UTKFace/
    - Aesthetic Score(AVA): https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment
    - Facial Beauty(SCUT_FBP5500_v2): https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores
    - Wild Image Quality(Koniq): https://database.mmsp-kn.de/koniq-10k-database.html
    - Modified Image Quality(Kadid): https://database.mmsp-kn.de/kadid-10k-database.html

### 2. Process Dataset
```bash
./script/process_data.sh
```

## LSD Training (DQN)
```bash
./script/train_dqn.sh
```