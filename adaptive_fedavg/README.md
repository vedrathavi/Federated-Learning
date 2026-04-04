# Adaptive FedAvg (IID, 4 Clients) — Pneumonia Detection

This folder contains a standalone **Adaptive FedAvg** implementation using a compact CNN for binary classification (`NORMAL` vs `PNEUMONIA`).

## What is fixed in this implementation

- **IID partitioning only**
- **Exactly 4 clients** (enforced by code)
- Windows/CPU-safe defaults (`num_workers=0`)
- All outputs saved in this folder under `outputs/`
- Previous outputs are auto-archived under `outputs_history/` before every new run

## Adaptive aggregation used

Client update weight per round is:

- data-size contribution (`beta_size`)
- client-performance contribution (`beta_perf`) from client shared-test accuracy

Final aggregation weight per client:

`adaptive_weight = beta_size * size_weight + beta_perf * performance_weight`

Where performance weights are temperature-scaled to emphasize better clients.

## Run

From workspace root:

```powershell
python adaptive_fedavg/adaptive_fedavg_pneumonia_cnn.py
```

## Current training setup

- Communication rounds: **20**
- Local epochs per client per round: **5**
- Model: **4-layer CNN**

## Expected dataset structure

Uses `dataset/` at workspace root:

```text
dataset/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Detailed outputs generated

Inside `adaptive_fedavg/outputs/`:

- `global_round_metrics.csv` — round-wise global metrics + confusion counts + drift
- `client_round_metrics.csv` — per-round per-client local train loss, local shared-test metrics, and adaptive weights
- `weight_drift.csv` — round-wise global model drift (L2)
- `per_client_results.csv` — final global model performance on each client local validation split
- `final_global_metrics.txt` and `final_global_metrics.json`
- `adaptive_fedavg_summary.json` — run config + summary statistics

Inside `adaptive_fedavg/outputs_history/`:

- `run_YYYYMMDD_HHMMSS/` — archived outputs from prior runs (CSV/JSON/TXT + plots)

Inside `adaptive_fedavg/outputs/plots/`:

- `global_metrics_vs_rounds.png`
- `client_accuracy_over_rounds.png`
- `client_convergence_effect.png`
- `final_accuracy_per_client_bar.png`
- `roc_curve_global.png` (when both classes exist in test labels)
