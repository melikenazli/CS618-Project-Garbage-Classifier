Project Execution Guide

1. split_data.py
- Splits the raw dataset into train/val/test folders using a fixed ratio (70/15/15).
- Run this once

2. train.py (Baseline / ResNet)
- Trains the selected model (set in config.py) from start to finish, saves the best checkpoint, and generates evaluation reports and confusion matrices.
- For Baseline CNN, set epoch number to 40 and model name to "baseline"
- For ResNet-50, set epoch number to 10 and model name to "resnet50"

3. train_efficientnet.py (EfficientNet two-phase fine-tuning)
- Performs the two-phase training procedure (frozen backbone and fine-tuning) specifically for EfficientNet-B0, producing best_efficientnet_b0_phase1.pth and best_efficientnet_b0_phase2.pth.

4. ensemble_eval_efficientnet_baseline.py
- Loads EfficientNet and Baseline CNN checkpoints, combines their logits with custom weights (0.9/0.1), and evaluates the ensemble on the test set.

5. ensemble_eval_efficientnet_resnet.py
- Performs equal-weight ensemble evaluation of EfficientNet-B0 and ResNet-50 using averaged logits.

6. test_interface.py (Gradio UI)
- Launches an interactive Gradio interface where users can upload/crop images and receive model predictions in real time.