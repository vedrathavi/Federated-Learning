# Federated Learning on Chest X-Ray Dataset

This project simulates federated learning across 4 hospitals for pneumonia detection using chest X-rays.

## Structure
- `dataset/`: contains train, val, and test folders.
- `models/`: CNN model for image classification.
- `utils/`: helper scripts for training, data loading, and evaluation.
- `main.py`: runs the full federated simulation.

## How it works
1. Each hospital trains locally on its private data.
2. The server collects and averages their weights.
3. A global model is updated and evaluated on the test set.

## Metrics
Accuracy, Precision, Recall, and F1-score are used to track performance.

## Requirements
```bash
pip install torch torchvision scikit-learn
