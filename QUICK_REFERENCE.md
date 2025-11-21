# Quick Reference - Federated Learning Metrics

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## 📊 Metrics Tracked

| Metric Category | What's Tracked | Where to Find |
|-----------------|----------------|---------------|
| **Global Model** | Accuracy, F1, Precision, Recall, AUC-ROC | `*_global_metrics.csv` |
| **Per-Client** | Individual client metrics + variance | `*_client_metrics.csv` |
| **Communication** | Bytes sent/received per round | `*_communication.csv` |
| **Progress** | Metrics over all rounds | All CSV files + plots |

## 📁 Output Files

```
logs/fedavg_pneumonia_4clients_TIMESTAMP/
├── CSV Files (4)
│   ├── *_global_metrics.csv    - Round-by-round global performance
│   ├── *_client_metrics.csv    - Per-client final performance
│   ├── *_communication.csv     - Communication overhead
│   └── *_config.json           - Experiment configuration
├── Summary Files (2)
│   ├── *_summary.json          - Machine-readable summary
│   └── *_summary.txt           - Human-readable report
└── plots/ (6 PNG files)
    ├── accuracy_vs_rounds.png              - Main convergence plot
    ├── all_metrics_vs_rounds.png           - All metrics over time
    ├── client_accuracies_final.png         - Client comparison
    ├── client_metrics_comparison.png       - Detailed client metrics
    ├── communication_overhead.png          - Per-round communication
    └── cumulative_communication.png        - Total communication cost
```

## 🎯 Key Results to Check

### 1. Global Performance (Final Round)
Look in `*_summary.txt` under "FINAL GLOBAL MODEL PERFORMANCE"
- **Target**: Accuracy > 0.85, AUC-ROC > 0.90

### 2. Client Variance
Look in `*_summary.txt` under "PER-CLIENT PERFORMANCE"
- **Low Variance** (< 0.01): Clients perform similarly
- **High Variance** (> 0.05): Significant heterogeneity

### 3. Convergence
View `accuracy_vs_rounds.png`
- **Good**: Steady increase then plateau
- **Bad**: Oscillating or decreasing

### 4. Communication Cost
Check `*_communication.csv`
- **Total Data**: Sum of all rounds
- **Per Round**: Should be consistent across rounds

## 🔧 Configuration Variables (main.py)

```python
NUM_CLIENTS = 4           # Fixed number of clients
ROUNDS = 15               # Communication rounds
EPOCHS_PER_CLIENT = 3     # Local training epochs
BATCH_SIZE = 16           # Batch size
SEED = 42                 # Random seed
```

## 📈 Reading CSV Files

### Python (Pandas)
```python
import pandas as pd

# Load global metrics
df = pd.read_csv('logs/.../fedavg_pneumonia_4clients_global_metrics.csv')
print(df[['round', 'accuracy', 'f1_score', 'auc_roc']])

# Load client metrics
clients = pd.read_csv('logs/.../fedavg_pneumonia_4clients_client_metrics.csv')
print(clients.groupby('client_id')['accuracy'].mean())

# Load communication
comm = pd.read_csv('logs/.../fedavg_pneumonia_4clients_communication.csv')
print(f"Total MB: {comm['total_mb'].sum():.2f}")
```

### Excel
Just open the CSV file directly in Excel!

## 🎨 Plot Descriptions

| Plot | Shows | Use For |
|------|-------|---------|
| `accuracy_vs_rounds.png` | Global accuracy over rounds | Check convergence |
| `all_metrics_vs_rounds.png` | All 5 metrics over rounds | Comprehensive view |
| `client_accuracies_final.png` | Final client accuracies | Client comparison |
| `client_metrics_comparison.png` | All metrics per client | Detailed analysis |
| `communication_overhead.png` | Data per round | Communication cost |
| `cumulative_communication.png` | Total accumulated data | Total cost |

## 🔍 Interpreting Results

### Excellent Results ✅
- Accuracy > 0.85
- F1 Score > 0.85
- AUC-ROC > 0.90
- Client variance < 0.01
- Steady convergence

### Good Results ✔️
- Accuracy > 0.80
- F1 Score > 0.80
- AUC-ROC > 0.85
- Client variance < 0.05
- Clear convergence

### Needs Improvement ⚠️
- Accuracy < 0.75
- High client variance (> 0.05)
- No clear convergence
- Unstable training

## 📊 Summary Report Structure

1. **Configuration**: All hyperparameters
2. **Final Global Metrics**: Test set performance
3. **Best Accuracy**: Highest accuracy achieved
4. **Client Performance**: Individual + statistics
5. **Communication**: Total overhead

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| No logs directory | Run `main.py` - created automatically |
| Import errors | Run `pip install -r requirements.txt` |
| Low accuracy | Increase ROUNDS or EPOCHS_PER_CLIENT |
| High client variance | Check data distribution |
| Out of memory | Reduce BATCH_SIZE |

## 📚 Documentation Files

- `METRICS_GUIDE.md`: Detailed metrics explanation
- `SETUP.md`: Installation instructions
- `UPDATE_SUMMARY.md`: Complete list of changes
- `QUICK_REFERENCE.md`: This file!

## 💡 Tips

1. **Always check the summary.txt first** - it has everything
2. **Compare plots** to see trends visually
3. **Use timestamps** to track multiple experiments
4. **Save good models** by noting the best round
5. **Monitor communication cost** for efficiency

## 🎯 Next Steps

1. Run the experiment: `python main.py`
2. Check `logs/.../summary.txt` for overview
3. View plots in `logs/.../plots/`
4. Analyze CSV files for detailed data
5. Adjust configuration and re-run

## 📞 Need Help?

- Check `METRICS_GUIDE.md` for metric definitions
- See `SETUP.md` for installation help
- Review `UPDATE_SUMMARY.md` for implementation details
