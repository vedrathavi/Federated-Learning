import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from typing import Dict, Tuple, List
import torch.nn.functional as F


def evaluate(model, dataloader, device, return_probs=False):
    """Basic evaluation function returning accuracy, precision, recall, f1."""
    model.eval()
    y_true, y_pred = [], []
    y_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if return_probs:
                y_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    if return_probs:
        return acc, prec, rec, f1, y_true, y_pred, y_probs
    return acc, prec, rec, f1


def evaluate_comprehensive(model, dataloader, device) -> Dict[str, float]:
    """Comprehensive evaluation including AUC-ROC.
    
    Returns:
        Dictionary with keys: accuracy, precision, recall, f1_score, auc_roc
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            # For binary classification, take probability of positive class
            y_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    # Calculate AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
    except ValueError:
        # Handle case where only one class is present
        metrics['auc_roc'] = 0.0
    
    return metrics


def evaluate_client(model, dataloader, device, client_id: int) -> Dict[str, float]:
    """Evaluate a single client's performance.
    
    Returns:
        Dictionary with client_id and all metrics
    """
    metrics = evaluate_comprehensive(model, dataloader, device)
    metrics['client_id'] = client_id
    metrics['num_samples'] = len(dataloader.dataset)
    return metrics


def compute_client_variance(client_metrics: List[Dict[str, float]], metric_key: str = 'accuracy') -> float:
    """Compute variance of a specific metric across clients.
    
    Args:
        client_metrics: List of metric dictionaries from each client
        metric_key: The metric to compute variance for (default: 'accuracy')
    
    Returns:
        Variance value
    """
    values = [m[metric_key] for m in client_metrics if metric_key in m]
    if len(values) < 2:
        return 0.0
    return float(np.var(values))
