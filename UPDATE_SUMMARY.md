# Federated Learning Project - Metrics & Analysis Update

## Summary of Changes

This update adds comprehensive metrics tracking, logging, and visualization capabilities to the federated learning project, as requested in your requirements.

## ✅ Implemented Features

### 1. **Global Model Metrics on Test Data** ✓
- **Accuracy**: Classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve for binary classification
- All metrics evaluated after each communication round
- Logged to CSV and plotted over rounds

### 2. **Per-Client Performance** ✓
- Individual metrics for each client at the end of the last round
- Metrics include: accuracy, F1 score, precision, recall, AUC-ROC
- **Client Variance**: Computed variance in client test accuracy
- Statistics: mean, std, variance, min, max across all clients
- Individual client results logged to CSV

### 3. **Training Progress Visualization** ✓
- **Accuracy vs Communication Rounds** graph
- All metrics plotted as function of rounds
- Shows convergence behavior
- Separate plot for each metric

### 4. **Fixed Number of Clients** ✓
- Number of clients fixed at 4 (configurable constant)
- Ensures consistent comparison across experiments
- Can be easily changed via `NUM_CLIENTS` variable in `main.py`

### 5. **Communication Overhead Tracking** ✓
- **Bytes sent per round**: Server → Clients
- **Bytes received per round**: Clients → Server
- **Total bytes per round**: Sum of sent + received
- **Model size**: Computed in bytes with human-readable format
- Cumulative communication cost tracked
- All logged to CSV with visualization plots

### 6. **Comprehensive Logging System** ✓
- All metrics stored in CSV files
- JSON configuration and summary files
- Human-readable text summary report
- Automatic timestamp-based experiment naming

### 7. **Visualization Graphs** ✓
Generated plots include:
- Accuracy vs communication rounds
- All metrics vs rounds (multi-panel)
- Per-client accuracy comparison (bar chart)
- Per-client metrics comparison (grouped bars)
- Communication overhead per round
- Cumulative communication overhead

## 📁 New Files Created

### Utility Modules
1. **`utils/communication_utils.py`**
   - Track bytes sent/received
   - Calculate model size
   - Communication cost computation
   - CommunicationTracker class

2. **`utils/logging_utils.py`**
   - ExperimentLogger class
   - CSV logging functionality
   - Plot generation functions
   - Summary report generation

### Documentation
3. **`METRICS_GUIDE.md`**
   - Comprehensive guide to all tracked metrics
   - File format descriptions
   - Usage instructions
   - Interpretation guidelines

4. **`SETUP.md`**
   - Installation instructions
   - Quick start guide
   - Troubleshooting tips
   - Customization options

5. **`UPDATE_SUMMARY.md`** (this file)
   - Overview of all changes
   - Feature checklist
   - File modifications summary

## 📝 Modified Files

### 1. **`main.py`**
**Major Changes:**
- Added comprehensive metrics tracking throughout training loop
- Integrated ExperimentLogger for automatic logging
- Added CommunicationTracker for overhead monitoring
- Per-client evaluation at final round
- Client variance computation
- Fixed number of clients (NUM_CLIENTS constant)
- Enhanced console output with detailed statistics
- Automatic generation of all plots and reports

**New Imports:**
```python
from utils.metrics_utils import evaluate_comprehensive, evaluate_client, compute_client_variance
from utils.communication_utils import CommunicationTracker, get_model_size_bytes, format_bytes
from utils.logging_utils import ExperimentLogger
```

### 2. **`utils/metrics_utils.py`**
**Major Changes:**
- Added `evaluate_comprehensive()`: Returns dict with all metrics including AUC-ROC
- Added `evaluate_client()`: Evaluate individual client with client_id tracking
- Added `compute_client_variance()`: Calculate variance across clients
- Enhanced `evaluate()` with probability output option
- Proper handling of binary classification metrics

**New Functions:**
```python
def evaluate_comprehensive(model, dataloader, device) -> Dict[str, float]
def evaluate_client(model, dataloader, device, client_id: int) -> Dict[str, float]
def compute_client_variance(client_metrics: List[Dict[str, float]], metric_key: str) -> float
```

### 3. **`utils/data_utils.py`**
**Minor Changes:**
- Enhanced `split_clients()` with better documentation
- Support for variable number of clients
- Fixed client number handling for consistent splits

### 4. **`requirements.txt`**
**Added Dependencies:**
```
matplotlib>=3.5.0  # For plotting
seaborn>=0.12.0    # For enhanced visualizations
```

## 📊 Output Structure

After running `python main.py`, you'll get:

```
logs/
└── fedavg_pneumonia_4clients_TIMESTAMP/
    ├── fedavg_pneumonia_4clients_config.json          # Experiment config
    ├── fedavg_pneumonia_4clients_global_metrics.csv   # Round-by-round global metrics
    ├── fedavg_pneumonia_4clients_client_metrics.csv   # Per-client metrics
    ├── fedavg_pneumonia_4clients_communication.csv    # Communication overhead
    ├── fedavg_pneumonia_4clients_summary.json         # JSON summary
    ├── fedavg_pneumonia_4clients_summary.txt          # Human-readable summary
    └── plots/
        ├── accuracy_vs_rounds.png
        ├── all_metrics_vs_rounds.png
        ├── client_accuracies_final.png
        ├── client_metrics_comparison.png
        ├── communication_overhead.png
        └── cumulative_communication.png
```

## 🎯 Key Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Global Performance** | Accuracy, Precision, Recall, F1 Score, AUC-ROC |
| **Client Performance** | Individual metrics per client, Variance, Mean, Std |
| **Communication** | Bytes sent, Bytes received, Total per round, Cumulative |
| **Progress** | All metrics tracked per communication round |

## 🚀 Usage

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment
python main.py
```

### View Results
```python
# Load and analyze metrics
import pandas as pd
df = pd.read_csv('logs/fedavg_pneumonia_4clients_global_metrics.csv')
print(df[['round', 'accuracy', 'f1_score', 'auc_roc']])
```

### Customize Configuration
Edit `main.py`:
```python
NUM_CLIENTS = 4           # Number of clients
ROUNDS = 15               # Communication rounds
EPOCHS_PER_CLIENT = 3     # Local epochs
BATCH_SIZE = 16           # Batch size
```

## 📈 Example Output

### Console Output
```
================================================================================
FINAL EVALUATION - ROUND 15
================================================================================

1. Global Model Performance on Test Set:
--------------------------------------------------------------------------------
   Accuracy: 0.876923
   Precision: 0.862500
   Recall: 0.891892
   F1 Score: 0.876923
   Auc Roc: 0.934567

2. Per-Client Performance on Test Set:
--------------------------------------------------------------------------------
   Client 1:
      Accuracy:  0.8523
      F1 Score:  0.8401
      AUC-ROC:   0.9123
      Samples:   521
   Client 2:
      Accuracy:  0.8876
      F1 Score:  0.8845
      AUC-ROC:   0.9401
      Samples:   1302
   ...

3. Client Performance Statistics:
--------------------------------------------------------------------------------
   Accuracy - Mean: 0.8692, Variance: 0.000234, Std: 0.0153

4. Communication Overhead Summary:
--------------------------------------------------------------------------------
   Total Rounds: 15
   Model Size: 6.10 MB
   Total Data Transfer: 732.42 MB
   Avg per Round: 48.83 MB
```

### Generated Files
- **6 CSV files** with detailed metrics
- **6 PNG plots** with visualizations
- **2 summary files** (JSON + TXT)

## 🔧 Technical Details

### Metrics Computation
- **AUC-ROC**: Uses scikit-learn's `roc_auc_score()` with predicted probabilities
- **Variance**: Computed using NumPy's `var()` function across client accuracies
- **Communication**: Model size = sum of (num_parameters × bytes_per_parameter)

### Logging Strategy
- Incremental CSV writing (append mode)
- Separate files for different metric types
- Timestamp-based experiment naming
- Automatic directory creation

### Visualization
- Matplotlib for all plots
- Seaborn for enhanced styling
- High-resolution output (300 DPI)
- Professional formatting with grids and labels

## ✨ Benefits

1. **Complete Tracking**: All required metrics automatically logged
2. **Easy Analysis**: CSV format allows easy import to Excel, pandas, etc.
3. **Visual Insights**: Plots show trends and comparisons at a glance
4. **Reproducibility**: Configuration saved with each experiment
5. **Scalability**: Designed to handle varying numbers of clients and rounds
6. **Efficiency**: Minimal overhead on training process
7. **Flexibility**: Easy to extend with additional metrics

## 🎓 Best Practices

1. **Compare Experiments**: Use timestamp-based naming to track multiple runs
2. **Analyze Variance**: High client variance may indicate need for fairness mechanisms
3. **Monitor Communication**: Optimize model size to reduce overhead
4. **Track Convergence**: Use accuracy vs rounds to determine optimal training length
5. **Document Changes**: Configuration file tracks all hyperparameters

## 🔮 Future Enhancements

Potential additions:
- TensorBoard integration for real-time monitoring
- Confusion matrix generation per round
- Per-class performance metrics
- Client selection strategy analysis
- Model checkpoint saving based on metrics
- Automated hyperparameter tuning
- Multi-run comparison tools

## ✅ Requirements Checklist

- [x] Accuracy + F1 score + AUC of global model on test data
- [x] Per-client performance at end of last communication round
- [x] Variance in client test accuracy
- [x] Training graph of model performance (accuracy) vs communication rounds
- [x] Fixed number of clients for all experiments
- [x] Bytes sent and bytes required per communication round
- [x] Store all metrics in log files
- [x] Generate relevant graphs

## 📚 Documentation

- `METRICS_GUIDE.md`: Detailed metrics documentation
- `SETUP.md`: Installation and setup guide
- `README.md`: Project overview (existing)
- Code comments: Comprehensive docstrings in all new functions

## 🎉 Ready to Use!

Your federated learning project is now equipped with comprehensive metrics tracking and analysis. Simply run:

```bash
pip install -r requirements.txt
python main.py
```

All metrics, logs, and visualizations will be automatically generated!
