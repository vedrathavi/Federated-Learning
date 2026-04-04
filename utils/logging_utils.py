"""
Utilities for logging metrics and generating visualizations for federated learning experiments.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import numpy as np
import torch

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ExperimentLogger:
    """Logger for federated learning experiments with visualization support."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        """Initialize the experiment logger.
        
        Args:
            log_dir: Directory to save logs and plots
            experiment_name: Name of the experiment (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Create subdirectories
        self.plots_dir = self.log_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize storage
        self.global_metrics = []
        self.client_metrics = []
        self.weight_drift = []
        self.communication_metrics = []
        self.config = {}
        
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration.
        
        Args:
            config: Dictionary containing experiment configuration
        """
        self.config = config
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def log_global_metrics(self, round_num: int, metrics: Dict[str, float]):
        """Log global model metrics for a round.
        
        Args:
            round_num: Communication round number
            metrics: Dictionary of metrics (accuracy, f1_score, auc_roc, etc.)
        """
        entry = {'round': round_num, **metrics}
        self.global_metrics.append(entry)
        
        # Append to CSV
        csv_file = self.log_dir / f"{self.experiment_name}_global_metrics.csv"
        self._append_to_csv(csv_file, entry)
    
    def log_client_metrics(self, round_num: int, client_id: int, metrics: Dict[str, float]):
        """Log individual client metrics.
        
        Args:
            round_num: Communication round number
            client_id: Client identifier
            metrics: Dictionary of metrics
        """
        entry = {'round': round_num, 'client_id': client_id, **metrics}
        self.client_metrics.append(entry)
        
        # Append to CSV
        csv_file = self.log_dir / f"{self.experiment_name}_client_metrics.csv"
        self._append_to_csv(csv_file, entry)

    def log_weight_drift(self, round_num: int, client_id: int, drift_value: float):
        """Log per-client weight drift (L2 distance) for a round."""
        entry = {'round': round_num, 'client_id': client_id, 'weight_drift': float(drift_value)}
        self.weight_drift.append(entry)
        csv_file = self.log_dir / f"{self.experiment_name}_weight_drift.csv"
        self._append_to_csv(csv_file, entry)
    
    def log_communication_metrics(self, round_num: int, metrics: Dict[str, Any]):
        """Log communication overhead metrics.
        
        Args:
            round_num: Communication round number
            metrics: Dictionary of communication metrics
        """
        entry = {'round': round_num, **metrics}
        self.communication_metrics.append(entry)
        
        # Append to CSV
        csv_file = self.log_dir / f"{self.experiment_name}_communication.csv"
        self._append_to_csv(csv_file, entry)
    
    def _append_to_csv(self, filepath: Path, data: Dict):
        """Append data to CSV file.
        
        Args:
            filepath: Path to CSV file
            data: Dictionary to write
        """
        file_exists = filepath.exists()
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    
    def plot_global_metrics(self):
        """Generate plots for global model performance over rounds."""
        if not self.global_metrics:
            return
        
        rounds = [m['round'] for m in self.global_metrics]
        
        # Plot accuracy over rounds
        if 'accuracy' in self.global_metrics[0]:
            plt.figure(figsize=(10, 6))
            accuracy = [m['accuracy'] for m in self.global_metrics]
            plt.plot(rounds, accuracy, marker='o', linewidth=2, markersize=6)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Global Model Accuracy vs Communication Rounds', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{self.experiment_name}_accuracy_vs_rounds.png", dpi=300)
            plt.close()
        
        # Plot all metrics in subplots
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc']
        available_metrics = [m for m in metrics_to_plot if m in self.global_metrics[0]]
        
        if len(available_metrics) > 1:
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(available_metrics):
                values = [m[metric] for m in self.global_metrics]
                axes[idx].plot(rounds, values, marker='o', linewidth=2, markersize=6)
                axes[idx].set_xlabel('Round', fontsize=10)
                axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                axes[idx].set_title(f'{metric.replace("_", " ").title()} vs Rounds', fontsize=11, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{self.experiment_name}_all_metrics_vs_rounds.png", dpi=300)
            plt.close()
    
    def plot_client_performance(self, final_round: int):
        """Generate plots for per-client performance at final round.
        
        Args:
            final_round: The final communication round number
        """
        if not self.client_metrics:
            return
        
        # Filter metrics for final round
        final_metrics = [m for m in self.client_metrics if m['round'] == final_round]
        if not final_metrics:
            return
        
        client_ids = [m['client_id'] for m in final_metrics]
        
        # Plot client accuracies
        if 'accuracy' in final_metrics[0]:
            plt.figure(figsize=(10, 6))
            accuracies = [m['accuracy'] for m in final_metrics]
            bars = plt.bar(client_ids, accuracies, color=sns.color_palette("husl", len(client_ids)))
            plt.xlabel('Client ID', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Per-Client Test Accuracy at Round {final_round}', fontsize=14, fontweight='bold')
            plt.xticks(client_ids)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.4f}',
                        ha='center', va='bottom', fontsize=9)
            
            # Add mean line
            mean_acc = np.mean(accuracies)
            plt.axhline(y=mean_acc, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{self.experiment_name}_client_accuracies_final.png", dpi=300)
            plt.close()
        
        # Plot comparison of all metrics across clients
        metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc']
        available_metrics = [m for m in metrics_to_plot if m in final_metrics[0]]
        
        if len(available_metrics) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(client_ids))
            width = 0.15
            
            for idx, metric in enumerate(available_metrics):
                values = [m[metric] for m in final_metrics]
                offset = width * (idx - len(available_metrics)/2)
                ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('Client ID', fontsize=12)
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_title(f'Per-Client Metrics Comparison at Round {final_round}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(client_ids)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{self.experiment_name}_client_metrics_comparison.png", dpi=300)
            plt.close()

    def plot_client_accuracy_over_rounds(self):
        """Plot per-round client accuracy (each client is a line across rounds)."""
        if not self.client_metrics:
            return

        # Gather rounds and client ids
        rounds = sorted(list({m['round'] for m in self.client_metrics}))
        client_ids = sorted(list({m['client_id'] for m in self.client_metrics}))

        # Build mapping client_id -> {round: accuracy}
        client_round_acc = {cid: {r: None for r in rounds} for cid in client_ids}
        for entry in self.client_metrics:
            cid = entry['client_id']
            rnd = entry['round']
            if 'accuracy' in entry:
                client_round_acc[cid][rnd] = entry['accuracy']

        plt.figure(figsize=(10, 6))
        for cid in client_ids:
            accs = [client_round_acc[cid].get(r, None) for r in rounds]
            # Replace None with np.nan so matplotlib doesn't break lines
            accs = [np.nan if v is None else v for v in accs]
            plt.plot(rounds, accs, marker='o', linewidth=2, markersize=6, label=f'Client {cid}')

        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Client Accuracy Across Rounds', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.experiment_name}_per_client_accuracy_vs_rounds.png", dpi=300)
        plt.close()

    def plot_weight_drift(self):
        """Plot Client Weight Drift vs Communication Rounds."""
        if not self.weight_drift:
            return

        rounds = sorted(list({m['round'] for m in self.weight_drift}))
        client_ids = sorted(list({m['client_id'] for m in self.weight_drift}))

        # Build mapping client_id -> {round: drift}
        client_round_drift = {cid: {r: None for r in rounds} for cid in client_ids}
        for entry in self.weight_drift:
            cid = entry['client_id']
            rnd = entry['round']
            client_round_drift[cid][rnd] = entry['weight_drift']

        plt.figure(figsize=(10, 6))
        for cid in client_ids:
            drifts = [client_round_drift[cid].get(r, None) for r in rounds]
            drifts = [np.nan if v is None else v for v in drifts]
            plt.plot(rounds, drifts, marker='o', linewidth=2, markersize=6, label=f'Client {cid}')

        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Weight Drift (L2)', fontsize=12)
        plt.title('Client Weight Drift vs Communication Rounds', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.experiment_name}_weight_drift_vs_rounds.png", dpi=300)
        plt.close()
    
    def plot_communication_costs(self):
        """Generate plots for communication overhead."""
        if not self.communication_metrics:
            return
        
        rounds = [m['round'] for m in self.communication_metrics]
        
        # Plot bytes per round
        if 'total_bytes' in self.communication_metrics[0]:
            plt.figure(figsize=(10, 6))
            total_bytes = [m['total_bytes'] / (1024**2) for m in self.communication_metrics]  # Convert to MB
            plt.plot(rounds, total_bytes, marker='s', linewidth=2, markersize=6, color='coral')
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Total Data Transfer (MB)', fontsize=12)
            plt.title('Communication Overhead per Round', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{self.experiment_name}_communication_overhead.png", dpi=300)
            plt.close()

    def plot_roc_curve(self, model, test_loader, device):
        """Plot and save ROC curve for binary classification using the provided model and test loader.

        Args:
            model: Trained PyTorch model
            test_loader: DataLoader for test set
            device: torch.device
        """
        try:
            from sklearn import metrics as skm
        except Exception:
            return

        if test_loader is None:
            return

        model.to(device)
        model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                # assume binary classification: take prob of positive class (index 1)
                if probs.shape[1] == 1:
                    scores = probs[:, 0]
                else:
                    scores = probs[:, 1]
                y_score.extend(scores.tolist())
                y_true.extend(labels.numpy().tolist())

        if len(set(y_true)) < 2:
            return

        fpr, tpr, _ = skm.roc_curve(y_true, y_score, pos_label=1)
        auc_val = skm.auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{self.experiment_name}_roc_curve.png", dpi=300)
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'summary': {}
        }
        
        # Global metrics summary
        if self.global_metrics:
            final_metrics = self.global_metrics[-1]
            report['summary']['final_global_metrics'] = final_metrics
            
            # Best metrics across all rounds
            if 'accuracy' in self.global_metrics[0]:
                best_acc_entry = max(self.global_metrics, key=lambda x: x['accuracy'])
                report['summary']['best_accuracy'] = {
                    'value': best_acc_entry['accuracy'],
                    'round': best_acc_entry['round']
                }
        
        # Client metrics summary
        if self.client_metrics:
            final_round = max(m['round'] for m in self.client_metrics)
            final_client_metrics = [m for m in self.client_metrics if m['round'] == final_round]
            
            if final_client_metrics and 'accuracy' in final_client_metrics[0]:
                accuracies = [m['accuracy'] for m in final_client_metrics]
                report['summary']['client_performance'] = {
                    'mean_accuracy': float(np.mean(accuracies)),
                    'variance_accuracy': float(np.var(accuracies)),
                    'std_accuracy': float(np.std(accuracies)),
                    'min_accuracy': float(np.min(accuracies)),
                    'max_accuracy': float(np.max(accuracies)),
                    'per_client': final_client_metrics
                }
        
        # Communication summary
        if self.communication_metrics:
            total_bytes = sum(m['total_bytes'] for m in self.communication_metrics)
            report['summary']['communication'] = {
                'total_bytes': total_bytes,
                'total_mb': total_bytes / (1024**2),
                'total_gb': total_bytes / (1024**3),
                'avg_bytes_per_round': total_bytes / len(self.communication_metrics),
                'total_rounds': len(self.communication_metrics)
            }
        
        # Save report
        report_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Also create a human-readable text report
        self._generate_text_report(report)
        
        return report
    
    def _generate_text_report(self, report: Dict):
        """Generate a human-readable text report.
        
        Args:
            report: Dictionary containing report data
        """
        report_file = self.log_dir / f"{self.experiment_name}_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"FEDERATED LEARNING EXPERIMENT REPORT\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration
            if report['config']:
                f.write("CONFIGURATION:\n")
                f.write("-" * 80 + "\n")
                for key, value in report['config'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Global metrics
            if 'final_global_metrics' in report['summary']:
                f.write("FINAL GLOBAL MODEL PERFORMANCE:\n")
                f.write("-" * 80 + "\n")
                for key, value in report['summary']['final_global_metrics'].items():
                    if key != 'round':
                        f.write(f"{key}: {value:.6f}\n")
                f.write("\n")
            
            # Best accuracy
            if 'best_accuracy' in report['summary']:
                best = report['summary']['best_accuracy']
                f.write(f"BEST ACCURACY: {best['value']:.6f} (achieved at round {best['round']})\n\n")
            
            # Client performance
            if 'client_performance' in report['summary']:
                cp = report['summary']['client_performance']
                f.write("PER-CLIENT PERFORMANCE (Final Round):\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mean Accuracy: {cp['mean_accuracy']:.6f}\n")
                f.write(f"Std Deviation: {cp['std_accuracy']:.6f}\n")
                f.write(f"Variance: {cp['variance_accuracy']:.6f}\n")
                f.write(f"Min Accuracy: {cp['min_accuracy']:.6f}\n")
                f.write(f"Max Accuracy: {cp['max_accuracy']:.6f}\n\n")
                
                f.write("Individual Client Results:\n")
                for client in cp['per_client']:
                    f.write(f"  Client {client['client_id']}: ")
                    f.write(f"acc={client.get('accuracy', 0):.4f}, ")
                    f.write(f"f1={client.get('f1_score', 0):.4f}, ")
                    f.write(f"auc={client.get('auc_roc', 0):.4f}\n")
                f.write("\n")
            
            # Communication
            if 'communication' in report['summary']:
                comm = report['summary']['communication']
                f.write("COMMUNICATION OVERHEAD:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Rounds: {comm['total_rounds']}\n")
                f.write(f"Total Data Transfer: {comm['total_mb']:.2f} MB ({comm['total_gb']:.4f} GB)\n")
                f.write(f"Avg per Round: {comm['avg_bytes_per_round'] / (1024**2):.2f} MB\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generation complete.\n")
