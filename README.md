# Garbage Classification Using Deep Learning  
A deep-learning-based image classifier for six waste categories: **cardboard, glass, metal, paper, plastic, and trash**.  
This project explores baseline CNNs, transfer learning (ResNet-50 & EfficientNet-B0), and ensemble learning to achieve high waste-sorting accuracy.

---

## Project Overview
The goal of this project is to build an automated garbage classification system that can assist recycling workflows and reduce human error in waste sorting.  
We compare multiple deep learning models:

- **Baseline CNN** — custom lightweight model trained from scratch  
- **ResNet-50** — pretrained ImageNet model with a fine-tuned final layer  
- **EfficientNet-B0** — pretrained ImageNet model fine-tuned using a two-phase strategy  
- **Ensembles** — combining model predictions for more reliable classification  

A Gradio-based UI is included for real-time predictions.

---

## Dataset
Two datasets were used and combined into a single 6-class dataset:

1. **CCHANG (Kaggle)**: Garbage Classification Dataset (https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)
2. **Suman Kunwar (Kaggle)**: Garbage Classification V2 (https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) 

Place raw data into:
- data/raw/

Then run the split script:
- python split_data.py

This generates an organized directory:
data/split/train/
data/split/val/
data/split/test/

## Model Architectures
1. Baseline CNN (Custom)
A 4-block CNN with batch normalization, ReLU, max-pooling, and dropout.
Lightweight (~0.42M parameters) and fast to train.

2. ResNet-50 (Transfer Learning)
Pretrained on ImageNet.
Only the final FC layer was retrained for 6 classes.

3. EfficientNet-B0 (Transfer Learning + Fine-Tuning)
Two-phase training:
Phase 1: Freeze all pretrained layers → train classifier head
Phase 2: Unfreeze last 2–3 blocks → fine-tune with low LR
EfficientNet achieved the best single-model accuracy.

4. Ensembles
We test two:
EfficientNet-B0 + Baseline CNN (weighted ensemble)
EfficientNet-B0 + ResNet-50 (equal-weight ensemble)
Ensembling improves stability and robustness.

# Running the project
1. Install dependencies
pip install -r requirements.txt
2. Split the dataset
python split_data.py
3. Train a model
Specify model in config.py
For Baseline and ResNet-50:
python train.py
For EfficientNetB0:
python train_efficientnet.py
4. Run ensemble evaluation
EfficientNet + Baseline:
python ensemble_eval_efficientnet_baseline.py

EfficientNet + ResNet:
python ensemble_eval_efficientnet_resnet.py
5. Launch Gradio Interface
python test_interface.py




