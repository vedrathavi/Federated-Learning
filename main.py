import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn_model import SimpleCNN
from utils.data_utils import load_datasets, split_clients, get_client_loaders,set_seed
from utils.train_utils import train_local
from utils.fed_avg import fed_avg
from utils.metrics_utils import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    set_seed(42)
    dataset, valset, testset = load_datasets("dataset")
    client_datasets = split_clients(dataset)
    client_loaders = get_client_loaders(client_datasets, num_workers=0)

    global_model = SimpleCNN().to(device)

    rounds = 3
    epochs_per_client = 3
    val_loader = DataLoader(valset, batch_size=16) if valset is not None else None

    for rnd in range(rounds):
        tqdm.write(f"\n--- Round {rnd+1} ---")
        local_weights = []

        for i, loader in enumerate(client_loaders):
            tqdm.write(f"Client {i+1} training (round {rnd+1})...")
            state_dict, history = train_local(global_model, loader, device, val_loader=val_loader, epochs=epochs_per_client, client_id=i+1)
            local_weights.append(state_dict)
            
            
        # aggregate and update global model using simple FedAvg (per round)
        global_model = fed_avg(global_model, local_weights)

        # quick evaluation on test set after this round
        acc, prec, rec, f1 = evaluate(global_model, DataLoader(testset, batch_size=16), device)
        tqdm.write(f"After round {rnd+1} test -> acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")


    # Final evaluation after all rounds
    tqdm.write("\n=== Final evaluation on test set ===")
    acc, prec, rec, f1 = evaluate(global_model, DataLoader(testset, batch_size=16), device)
    tqdm.write(f"Final test -> acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

if __name__ == "__main__":
    main()
