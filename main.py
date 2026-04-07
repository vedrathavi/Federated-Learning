import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import resnet18
try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None
import torch.nn as nn
from utils.data_utils import load_datasets, split_clients, get_client_loaders, set_seed
from utils.train_utils import train_local
from utils.fed_avg import fed_avg
from utils.metrics_utils import evaluate_comprehensive, evaluate_client, compute_client_variance
from utils.communication_utils import CommunicationTracker, get_model_size_bytes, format_bytes, state_dict_l2_distance
from copy import deepcopy
from utils.logging_utils import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg training for pneumonia detection")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Dataset root containing train/val/test folders",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs/metrics/plots are saved",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix appended to experiment name (e.g., dataset name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed hyperparameters
    SEED = 42
    NUM_CLIENTS = 4  # Fixed number of clients
    ROUNDS = 10
    EPOCHS_PER_CLIENT = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4  # Learning rate for local training
    DIRICHLET_ALPHA = 0.5  # Dirichlet concentration for non-IID partition
    
    set_seed(SEED)

    data_dir = os.path.abspath(args.data_dir)
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    experiment_name = f"fedavg_pneumonia_{NUM_CLIENTS}clients"
    if args.run_tag.strip():
        experiment_name = f"{experiment_name}_{args.run_tag.strip()}"
    
    logger = ExperimentLogger(log_dir=log_dir, experiment_name=experiment_name)
    config = {
        'device': str(device), 'seed': SEED, 'num_clients': NUM_CLIENTS,
        'rounds': ROUNDS, 'epochs_per_client': EPOCHS_PER_CLIENT,
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'model': 'ResNet-18', 'aggregation': 'FedAvg', 'dataset': 'Pneumonia X-Ray',
        'dataset_path': data_dir, 'log_dir': log_dir
    }
    logger.log_config(config)
    print(f"Experiment: {logger.experiment_name} — logs: {logger.log_dir}")
    print(f"Dataset path: {data_dir}")

    trainset, valset, testset = load_datasets(data_dir, img_size=224, to_3ch=True)
    client_datasets = split_clients(trainset, num_clients=NUM_CLIENTS, partition='dirichlet', alpha=DIRICHLET_ALPHA)
    client_loaders = get_client_loaders(client_datasets, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(trainset.classes)
    # Use weights enum when available to avoid deprecation warning
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        global_model = resnet18(weights=weights)
    else:
        global_model = resnet18(pretrained=True)
    global_model.fc = nn.Linear(global_model.fc.in_features, num_classes)
    global_model.to(device)
    
    # Initialize communication tracker
    comm_tracker = CommunicationTracker()
    model_size = get_model_size_bytes(global_model)
    print(f"Model size: {format_bytes(model_size)}")
    
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE) if valset is not None else None

    # Training loop
    print(f"Starting training: {NUM_CLIENTS} clients, {ROUNDS} rounds")
    
    for rnd in range(ROUNDS):
        tqdm.write(f"Round {rnd+1}/{ROUNDS}")
        local_weights = []

        # Local training phase
        for i, loader in enumerate(client_loaders):
            tqdm.write(f"Client {i+1}/{NUM_CLIENTS} training...")
            # Capture global weights before local update for drift computation
            global_state_before = deepcopy(global_model.state_dict())

            state_dict, history = train_local(
                global_model,
                loader,
                device,
                val_loader=val_loader,
                epochs=EPOCHS_PER_CLIENT,
                lr=LEARNING_RATE,
                client_id=i+1
            )

            # Compute weight drift (L2 distance) between local update and global model
            drift = state_dict_l2_distance(state_dict, global_state_before)
            logger.log_weight_drift(round_num=rnd + 1, client_id=i+1, drift_value=drift)

            local_weights.append(state_dict)
        
        # Track communication cost
        comm_tracker.add_round(rnd + 1, NUM_CLIENTS, model_size)
        comm_cost = comm_tracker.round_costs[-1]
        logger.log_communication_metrics(
            round_num=rnd + 1,
            metrics={
                'bytes_sent_to_clients': comm_cost['bytes_sent_to_clients'],
                'bytes_received_from_clients': comm_cost['bytes_received_from_clients'],
                'total_bytes': comm_cost['total_bytes'],
                'total_mb': comm_cost['total_bytes'] / (1024**2),
                'model_size_bytes': model_size,
                'num_clients': NUM_CLIENTS
            }
        )
        
        # Aggregate and update global model using FedAvg
        global_model = fed_avg(global_model, local_weights)

        # Evaluate global model on test set
        global_metrics = evaluate_comprehensive(global_model, test_loader, device)
        logger.log_global_metrics(round_num=rnd + 1, metrics=global_metrics)
        # Print concise global metrics for monitoring
        tqdm.write(f"Round {rnd+1} global -> acc: {global_metrics['accuracy']:.4f}, f1: {global_metrics['f1_score']:.4f}, auc: {global_metrics['auc_roc']:.4f}")

        # Evaluate and log per-client performance for this round (global model on each client's local data)
        for i, loader in enumerate(client_loaders):
            client_metrics = evaluate_client(global_model, loader, device, client_id=i+1)
            logger.log_client_metrics(round_num=rnd + 1, client_id=i+1, metrics=client_metrics)

        # Log communication for this round (concise)
        tqdm.write(f"Comm: {format_bytes(comm_cost['total_bytes'])}")

    # ========================================================================
    # Final comprehensive evaluation
    # ========================================================================
    # Final evaluation and summary
    final_global_metrics = evaluate_comprehensive(global_model, test_loader, device)
    print("Final global metrics:", ", ".join([f"{k}={v:.4f}" for k,v in final_global_metrics.items()]))

    final_round = ROUNDS
    final_client_metrics = [m for m in logger.client_metrics if m['round'] == final_round]
    client_accuracies = [m['accuracy'] for m in final_client_metrics]
    mean_acc = float(sum(client_accuracies)/len(client_accuracies)) if client_accuracies else 0.0
    accuracy_variance = compute_client_variance(final_client_metrics, 'accuracy')
    print(f"Per-client mean accuracy: {mean_acc:.4f}, variance: {accuracy_variance:.6f}")
    
    # ========================================================================
    # Generate visualizations and reports
    # ========================================================================
    # Generate plots and summary
    logger.plot_global_metrics()
    logger.plot_client_performance(final_round=ROUNDS)
    logger.plot_client_accuracy_over_rounds()
    logger.plot_communication_costs()
    # Final ROC curve for global model
    logger.plot_roc_curve(global_model, test_loader, device)
    summary_report = logger.generate_summary_report()

    print(f"Experiment complete. Results saved to: {logger.log_dir}")


if __name__ == "__main__":
    main()
