# Federated Learning Metrics & Analysis Guide

## Overview

This project now includes comprehensive metrics tracking, logging, and visualization for federated learning experiments. All metrics are automatically logged to CSV files and visualized through various plots.

## Tracked Metrics

### 1. Global Model Performance (Test Set)
Evaluated after each communication round:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve

### 2. Per-Client Performance
Evaluated at the end of the last communication round:
- Individual metrics for each client (Accuracy, F1, Precision, Recall, AUC-ROC)
- Number of samples per client
- **Variance in client test accuracy**: Measures heterogeneity in client performance
- Mean, standard deviation, min, and max accuracy across clients

### 3. Training Progress
- Global model accuracy tracked as a function of communication rounds
- All metrics plotted over rounds to visualize convergence

### 4. Communication Overhead
- **Bytes sent to clients**: Data transfer from server to clients per round
- **Bytes received from clients**: Data transfer from clients to server per round
- **Total bytes per round**: Sum of sent and received data
- **Model size**: Size of the model parameters in bytes
- **Cumulative communication cost**: Total data transfer over all rounds

### 5. Fixed Number of Clients
- The number of clients is fixed at 4 (configurable via `NUM_CLIENTS` in `main.py`)
- Ensures consistent comparisons across different aggregation schemes

## Output Files

### Log Directory Structure
```
logs/
├── fedavg_pneumonia_4clients_config.json          # Experiment configuration
├── fedavg_pneumonia_4clients_global_metrics.csv   # Global model metrics per round
├── fedavg_pneumonia_4clients_client_metrics.csv   # Per-client metrics
├── fedavg_pneumonia_4clients_communication.csv    # Communication overhead per round
├── fedavg_pneumonia_4clients_summary.json         # Comprehensive summary (JSON)
├── fedavg_pneumonia_4clients_summary.txt          # Human-readable summary report
└── plots/
    ├── fedavg_pneumonia_4clients_accuracy_vs_rounds.png
    ├── fedavg_pneumonia_4clients_all_metrics_vs_rounds.png
    ├── fedavg_pneumonia_4clients_client_accuracies_final.png
    ├── fedavg_pneumonia_4clients_client_metrics_comparison.png
    ├── fedavg_pneumonia_4clients_communication_overhead.png
    └── fedavg_pneumonia_4clients_cumulative_communication.png
```

### CSV Files

#### 1. `global_metrics.csv`
| round | accuracy | precision | recall | f1_score | auc_roc |
|-------|----------|-----------|--------|----------|---------|
| 1     | 0.8234   | 0.8156    | 0.8423 | 0.8287   | 0.8912  |
| 2     | 0.8567   | 0.8445    | 0.8712 | 0.8576   | 0.9123  |
| ...   | ...      | ...       | ...    | ...      | ...     |

#### 2. `client_metrics.csv`
| round | client_id | accuracy | precision | recall | f1_score | auc_roc | num_samples |
|-------|-----------|----------|-----------|--------|----------|---------|-------------|
| 15    | 1         | 0.8456   | 0.8234    | 0.8678 | 0.8451   | 0.9034  | 521         |
| 15    | 2         | 0.8723   | 0.8567    | 0.8912 | 0.8736   | 0.9234  | 1302        |
| ...   | ...       | ...      | ...       | ...    | ...      | ...     | ...         |

#### 3. `communication.csv`
| round | bytes_sent_to_clients | bytes_received_from_clients | total_bytes | total_mb | model_size_bytes | num_clients |
|-------|-----------------------|-----------------------------|-------------|----------|------------------|-------------|
| 1     | 25600000              | 25600000                    | 51200000    | 48.83    | 6400000          | 4           |
| 2     | 25600000              | 25600000                    | 51200000    | 48.83    | 6400000          | 4           |
| ...   | ...                   | ...                         | ...         | ...      | ...              | ...         |

## Generated Plots

### 1. Accuracy vs Communication Rounds
Shows global model accuracy improvement over training rounds.

### 2. All Metrics vs Rounds
Multi-panel plot showing accuracy, F1 score, precision, recall, and AUC-ROC over rounds.

### 3. Per-Client Test Accuracy (Final Round)
Bar chart comparing final accuracy across all clients with mean line.

### 4. Client Metrics Comparison (Final Round)
Grouped bar chart comparing all metrics across clients.

### 5. Communication Overhead per Round
Line plot showing data transfer (in MB) for each communication round.

### 6. Cumulative Communication Overhead
Line plot showing total accumulated data transfer over all rounds.

## Summary Report

The `summary.txt` file contains:
- Experiment configuration
- Final global model performance
- Best accuracy achieved and at which round
- Per-client performance statistics (mean, variance, std, min, max)
- Individual client results
- Total communication overhead

## Usage

### Running an Experiment

```python
python main.py
```

The script will:
1. Train the federated learning model
2. Track all metrics automatically
3. Generate CSV logs incrementally
4. Create all visualization plots
5. Generate summary reports

### Customizing Configuration

Edit `main.py` to change:
```python
NUM_CLIENTS = 4           # Number of federated clients
ROUNDS = 15               # Communication rounds
EPOCHS_PER_CLIENT = 3     # Local epochs per round
BATCH_SIZE = 16           # Batch size for training
SEED = 42                 # Random seed for reproducibility
```

### Accessing Logged Data

All experiments are saved in the `logs/` directory with timestamps. You can:
- Load CSV files for custom analysis
- View plots in the `plots/` subdirectory
- Read the summary report for quick insights

## Interpreting Results

### High Client Variance
- Indicates heterogeneous data distribution or client capabilities
- May suggest need for personalized models or fairness constraints

### Communication Overhead
- Total data transfer should scale linearly with rounds
- Compare different aggregation schemes by communication efficiency

### Convergence Analysis
- Look for plateau in accuracy vs rounds plot
- Early stopping can be applied if accuracy stops improving

## New Utility Modules

### `utils/metrics_utils.py`
- `evaluate_comprehensive()`: Full evaluation with AUC-ROC
- `evaluate_client()`: Per-client evaluation
- `compute_client_variance()`: Calculate metric variance across clients

### `utils/communication_utils.py`
- `CommunicationTracker`: Track bytes sent/received per round
- `get_model_size_bytes()`: Calculate model size
- `format_bytes()`: Human-readable byte formatting

### `utils/logging_utils.py`
- `ExperimentLogger`: Comprehensive logging and visualization
- Automatic CSV writing
- Plot generation
- Summary report creation

## Dependencies

Required packages (already in `requirements.txt`):
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.12.0` - Enhanced visualizations
- `scikit-learn>=1.2.2` - Metrics calculation (includes AUC-ROC)

Install with:
```bash
pip install -r requirements.txt
```

## Future Extensions

Potential additions to metrics tracking:
- Per-class performance metrics
- Confusion matrices per round
- Learning rate scheduling logs
- Client selection strategies comparison
- Model checkpoint saving based on best metrics
- Real-time dashboard (e.g., using TensorBoard)
- Comparison across multiple runs/configurations

## Notes

- All metrics are computed on the **test set** to ensure unbiased evaluation
- Client metrics use the global model evaluated on each client's local data
- Communication costs assume full model parameter transfer (can be optimized with compression)
- Fixed client number ensures fair comparison across different aggregation schemes
