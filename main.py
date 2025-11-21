import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn_model import SimpleCNN
from utils.data_utils import load_datasets, split_clients, get_client_loaders, set_seed
from utils.train_utils import train_local
from utils.fed_avg import fed_avg
from utils.metrics_utils import evaluate, evaluate_comprehensive, evaluate_client, compute_client_variance
from utils.communication_utils import CommunicationTracker, get_model_size_bytes, format_bytes
from utils.logging_utils import ExperimentLogger


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Fixed hyperparameters
    SEED = 42
    NUM_CLIENTS = 4  # Fixed number of clients
    ROUNDS = 30
    EPOCHS_PER_CLIENT = 2
    BATCH_SIZE = 16
    
    set_seed(SEED)
    
    # Initialize experiment logger
    logger = ExperimentLogger(log_dir="logs", experiment_name=f"fedavg_pneumonia_{NUM_CLIENTS}clients")
    
    # Log configuration
    config = {
        'device': str(device),
        'seed': SEED,
        'num_clients': NUM_CLIENTS,
        'rounds': ROUNDS,
        'epochs_per_client': EPOCHS_PER_CLIENT,
        'batch_size': BATCH_SIZE,
        'model': 'SimpleCNN',
        'aggregation': 'FedAvg',
        'dataset': 'Pneumonia X-Ray'
    }
    logger.log_config(config)
    print(f"\nExperiment: {logger.experiment_name}")
    print(f"Logs will be saved to: {logger.log_dir}")

    # Load datasets
    dataset, valset, testset = load_datasets("dataset")
    client_datasets = split_clients(dataset, num_clients=NUM_CLIENTS)
    client_loaders = get_client_loaders(client_datasets, batch_size=BATCH_SIZE, num_workers=0)
    
    # Create test loader for each client (for per-client evaluation)
    # We'll use a portion of test set or create client-specific test sets
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize global model
    global_model = SimpleCNN().to(device)
    
    # Initialize communication tracker
    comm_tracker = CommunicationTracker()
    model_size = get_model_size_bytes(global_model)
    print(f"\nModel size: {format_bytes(model_size)}")
    
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE) if valset is not None else None

    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting Federated Learning with {NUM_CLIENTS} clients for {ROUNDS} rounds")
    print(f"{'='*80}\n")
    
    for rnd in range(ROUNDS):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Round {rnd+1}/{ROUNDS}")
        tqdm.write(f"{'='*60}")
        
        local_weights = []

        # Local training phase
        for i, loader in enumerate(client_loaders):
            tqdm.write(f"Client {i+1}/{NUM_CLIENTS} training...")
            state_dict, history = train_local(
                global_model, 
                loader, 
                device, 
                val_loader=val_loader, 
                epochs=EPOCHS_PER_CLIENT, 
                client_id=i+1
            )
            local_weights.append(state_dict)
        
        # Track communication cost for this round
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
        
        tqdm.write(f"\nGlobal model performance (Round {rnd+1}):")
        tqdm.write(f"  Accuracy:  {global_metrics['accuracy']:.4f}")
        tqdm.write(f"  Precision: {global_metrics['precision']:.4f}")
        tqdm.write(f"  Recall:    {global_metrics['recall']:.4f}")
        tqdm.write(f"  F1 Score:  {global_metrics['f1_score']:.4f}")
        tqdm.write(f"  AUC-ROC:   {global_metrics['auc_roc']:.4f}")
        
        # Log communication for this round
        tqdm.write(f"\nCommunication overhead:")
        tqdm.write(f"  Total: {format_bytes(comm_cost['total_bytes'])}")

    # ========================================================================
    # Final comprehensive evaluation
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION - ROUND {ROUNDS}")
    print(f"{'='*80}\n")
    
    # 1. Global model evaluation on test set
    print("1. Global Model Performance on Test Set:")
    print("-" * 80)
    final_global_metrics = evaluate_comprehensive(global_model, test_loader, device)
    for metric, value in final_global_metrics.items():
        print(f"   {metric.replace('_', ' ').title()}: {value:.6f}")
    
    # 2. Per-client evaluation on test set
    print(f"\n2. Per-Client Performance on Test Set:")
    print("-" * 80)
    client_final_metrics = []
    client_accuracies = []
    
    for i, loader in enumerate(client_loaders):
        # Evaluate each client's local data with global model
        client_metrics = evaluate_client(global_model, loader, device, client_id=i+1)
        client_final_metrics.append(client_metrics)
        client_accuracies.append(client_metrics['accuracy'])
        
        # Log to logger
        logger.log_client_metrics(
            round_num=ROUNDS,
            client_id=i+1,
            metrics=client_metrics
        )
        
        print(f"   Client {i+1}:")
        print(f"      Accuracy:  {client_metrics['accuracy']:.4f}")
        print(f"      F1 Score:  {client_metrics['f1_score']:.4f}")
        print(f"      AUC-ROC:   {client_metrics['auc_roc']:.4f}")
        print(f"      Samples:   {client_metrics['num_samples']}")
    
    # 3. Client performance variance
    print(f"\n3. Client Performance Statistics:")
    print("-" * 80)
    accuracy_variance = compute_client_variance(client_final_metrics, 'accuracy')
    f1_variance = compute_client_variance(client_final_metrics, 'f1_score')
    
    print(f"   Accuracy - Mean: {sum(client_accuracies)/len(client_accuracies):.4f}, "
          f"Variance: {accuracy_variance:.6f}, Std: {accuracy_variance**0.5:.6f}")
    print(f"   F1 Score - Variance: {f1_variance:.6f}")
    
    # 4. Communication summary
    print(f"\n4. Communication Overhead Summary:")
    print("-" * 80)
    comm_summary = comm_tracker.get_summary()
    print(f"   Total Rounds: {comm_summary['total_rounds']}")
    print(f"   Model Size: {comm_summary['model_size_formatted']}")
    print(f"   Total Data Transfer: {comm_summary['total_bytes_formatted']}")
    print(f"   Avg per Round: {comm_summary['avg_bytes_per_round_formatted']}")
    
    # ========================================================================
    # Generate visualizations and reports
    # ========================================================================
    print(f"\n{'='*80}")
    print("Generating Visualizations and Reports...")
    print(f"{'='*80}\n")
    
    logger.plot_global_metrics()
    print("✓ Generated global metrics plots")
    
    logger.plot_client_performance(final_round=ROUNDS)
    print("✓ Generated per-client performance plots")
    
    logger.plot_communication_costs()
    print("✓ Generated communication overhead plots")
    
    summary_report = logger.generate_summary_report()
    print("✓ Generated summary report")
    
    print(f"\n{'='*80}")
    print(f"Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {logger.log_dir}")
    print(f"  - Metrics CSVs: {logger.log_dir}")
    print(f"  - Plots: {logger.plots_dir}")
    print(f"  - Summary: {logger.log_dir}/{logger.experiment_name}_summary.txt")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
