# Setup Instructions for Federated Learning Project

## Installation

### 1. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install scikit-learn>=1.2.2
pip install tqdm>=4.65.0
pip install numpy>=1.23.0
pip install Pillow>=9.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.12.0
```

### 2. Verify Dataset Structure

Ensure your dataset is organized as follows:

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Run the Experiment

```bash
python main.py
```

## What Gets Created

After running, you'll see:

1. **logs/** directory with:
   - CSV files for all metrics
   - JSON configuration and summary files
   - Text summary report

2. **logs/plots/** directory with:
   - Accuracy vs rounds plot
   - All metrics vs rounds plot
   - Per-client performance plots
   - Communication overhead plots

## Quick Start Example

```python
# Run with default settings (4 clients, 15 rounds)
python main.py
```

Expected output structure:
```
logs/
└── fedavg_pneumonia_4clients_YYYYMMDD_HHMMSS/
    ├── *_config.json
    ├── *_global_metrics.csv
    ├── *_client_metrics.csv
    ├── *_communication.csv
    ├── *_summary.json
    ├── *_summary.txt
    └── plots/
        ├── *_accuracy_vs_rounds.png
        ├── *_all_metrics_vs_rounds.png
        ├── *_client_accuracies_final.png
        ├── *_client_metrics_comparison.png
        ├── *_communication_overhead.png
        └── *_cumulative_communication.png
```

## Customization

Edit `main.py` to modify:
- `NUM_CLIENTS`: Number of federated clients (default: 4)
- `ROUNDS`: Number of communication rounds (default: 15)
- `EPOCHS_PER_CLIENT`: Local training epochs (default: 3)
- `BATCH_SIZE`: Batch size for training (default: 16)

## Troubleshooting

### Import Errors
If you get import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues
The code automatically detects and uses GPU if available. If you have GPU issues:
```python
# Force CPU usage by modifying main.py:
device = torch.device("cpu")
```

### Dataset Not Found
Ensure the `dataset` directory is in the same folder as `main.py` and contains the required subdirectories.

### Out of Memory
Reduce batch size in `main.py`:
```python
BATCH_SIZE = 8  # or smaller
```

## Viewing Results

### CSV Files
Open with Excel, pandas, or any CSV reader:
```python
import pandas as pd
df = pd.read_csv('logs/fedavg_pneumonia_4clients_global_metrics.csv')
print(df)
```

### Plots
View PNG files in the `logs/plots/` directory with any image viewer.

### Summary Report
Open `*_summary.txt` with any text editor for a comprehensive overview.

## Next Steps

1. **Compare Aggregation Schemes**: Modify `fed_avg.py` to implement different aggregation methods
2. **Experiment with Hyperparameters**: Change rounds, epochs, batch size
3. **Analyze Results**: Use generated CSV files for custom analysis
4. **Extend Metrics**: Add more metrics in `utils/metrics_utils.py`

## Additional Resources

- See `METRICS_GUIDE.md` for detailed metrics documentation
- Check `README.md` for project overview
- Review utility modules in `utils/` for implementation details
