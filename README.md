# Federated Learning (FedAvg) — Pneumonia detection (Chest X‑Rays)

This repository is a compact, runnable FedAvg simulation that trains a small CNN across multiple simulated clients (hospitals) using an ImageFolder-structured chest X‑ray dataset. It is intended for experimentation and learning, not for clinical use.

## Contents

- `main.py` — orchestrates dataset loading, client partitioning, local training and aggregation.
- `models/cnn_model.py` — simple CNN (uses adaptive pooling so input size is flexible).
- `utils/data_utils.py` — dataset transforms, loading, and client splitting.
- `utils/train_utils.py` — `train_local()` which trains a fresh local copy of the global model and returns its weights and per-epoch history.
- `utils/fed_avg.py` — simple (uniform) FedAvg aggregator. Averaging is device-safe (done on CPU).
- `utils/metrics_utils.py` — evaluation (accuracy, precision, recall, F1) using sklearn with safe defaults.

## Dataset layout

The repository expects a folder called `dataset/` with the typical ImageFolder structure:

```
dataset/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Quickstart

1. Install dependencies. Pick the correct PyTorch build for your system (CPU or CUDA) from https://pytorch.org/ then install other packages:

```powershell
# Example (CPU):
python -m pip install --user torch torchvision scikit-learn tqdm
```

2. Run the simulation:

```powershell
python main.py
```

## What the code does (step-by-step)

1. Set deterministic seeds via `utils/data_utils.set_seed(42)` for reproducible behavior.
2. Load the ImageFolder datasets (`train`, `val`, `test`) with grayscale transforms and normalization.
3. Split the training dataset into N client subsets (default 4) with user-defined proportions.
4. For a configured number of communication rounds:
   - Each client trains locally for a small number of epochs using `train_local()` which:
     - creates a fresh copy of the global model,
     - trains on the client DataLoader (shows a per-epoch tqdm progress bar with live postfix metrics),
     - returns the trained `state_dict` and a `history` list of per-epoch metrics.
   - The server aggregates client weights using `utils/fed_avg.fed_avg` (uniform average). Aggregation is done on CPU, then the averaged state is loaded back onto the model device.
   - The server evaluates the aggregated global model on the `test` set and logs accuracy / precision / recall / F1.

## Design decisions and correctness notes

- Aggregation: we use a simple uniform FedAvg (equal weights). This is implemented safely: client tensors are moved to CPU before averaging to avoid device mismatches and high GPU memory usage.
- Metrics: sklearn metrics are called with `zero_division=0` to avoid runtime warnings when a class is absent in predictions.
- Model: `SimpleCNN` uses `AdaptiveAvgPool2d((1,1))` so the FC layer doesn't rely on a hard-coded flatten size.
- Logging: per-batch progress bars are shown inside `train_local`. Higher-level headers use `tqdm.write()` to avoid corrupting active progress bars.
- Reproducibility: `set_seed()` configures Python/NumPy/PyTorch seeds and cuDNN deterministic flags (useful for development, but may slow training).

## Troubleshooting

- If you see duplicated output or strange prints on Windows terminals, run with `num_workers=0` (the project default) and avoid nested tqdm bars. The code already sets `num_workers=0` by default.
- If you get CUDA out-of-memory errors, either reduce batch size or run on CPU by installing a CPU-only PyTorch build.

## License / Notes

This code is provided as-is for research and learning. Not intended for clinical use. Modify, improve and extend as you like.


