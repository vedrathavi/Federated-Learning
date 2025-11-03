"""
fedavg_pneumonia_refactor.py

Refactored FedAvg simulation for Pneumonia (ImageFolder).
Single-file for convenience; split into modules when you want.

Key improvements vs original:
- Explicit grayscale->3ch handling (optional)
- Adaptive pooling instead of fixed fc flatten dims
- Client and Server classes for modularity
- Aggregator supports 'sample_weight' or 'uniform'
- Device-safe aggregation (CPU)
- CLI via argparse, basic CSV logging, tqdm progress
- Hooks for extensions (FedProx, secure agg) are trivial to add
"""

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

# -------------------------
# Config & Logging
# -------------------------
LOG = logging.getLogger("fedavg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_cpu_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.cpu().clone() for k, v in state.items()}

# -------------------------
# Model factory
# -------------------------
def get_simple_cnn(num_classes: int = 2, in_channels: int = 3) -> nn.Module:
    """Simple CNN with adaptive pooling so FC size is not hard-coded."""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes, in_channels):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),  # <- avoids hard-coded flatten size
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return SimpleCNN(num_classes, in_channels)

def get_resnet18(num_classes: int = 2, pretrained: bool = False, in_channels: int = 3) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    if in_channels != 3:
        # replace first conv to accept different #channels
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------
# Dataset helpers
# -------------------------
def get_transforms(img_size: int = 224, to_3ch: bool = True):
    """Return train / eval transforms. If dataset is grayscale and you want 3 channels,
    set to_3ch=True (converts to 3 identical channels)."""
    train_aug = []
    if to_3ch:
        train_aug.append(transforms.Grayscale(num_output_channels=3))
    train_aug += [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    eval_aug = []
    if to_3ch:
        eval_aug.append(transforms.Grayscale(num_output_channels=3))
    eval_aug += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(train_aug), transforms.Compose(eval_aug)

def load_imagefolder(root: Path, train_transform, eval_transform):
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform) if val_dir.exists() else None
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform) if test_dir.exists() else None
    return train_ds, val_ds, test_ds

# -------------------------
# Partitioners
# -------------------------
def iid_partition(dataset: Dataset, num_clients: int, proportions: Optional[List[float]] = None) -> Dict[int, List[int]]:
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    if proportions is None:
        # equal-ish splits
        sizes = [n // num_clients] * num_clients
        for i in range(n % num_clients):
            sizes[i] += 1
    else:
        # proportions must sum to 1
        sizes = [int(p * n) for p in proportions]
        # fix rounding leftover
        while sum(sizes) < n:
            sizes[np.argmax(proportions)] += 1
    partitions = {}
    ptr = 0
    for i, s in enumerate(sizes):
        partitions[i] = indices[ptr:ptr+s].tolist()
        ptr += s
    return partitions

def dirichlet_partition(dataset: Dataset, num_clients: int, alpha: float = 0.5) -> Dict[int, List[int]]:
    labels = np.array([dataset.samples[i][1] for i in range(len(dataset))])
    classes = np.unique(labels)
    client_idx = {i: [] for i in range(num_clients)}
    for c in classes:
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        proportions = np.random.dirichlet(alpha=[alpha]*num_clients)
        counts = (proportions * len(c_idx)).astype(int)
        while counts.sum() < len(c_idx):
            counts[np.argmax(proportions)] += 1
        ptr = 0
        for i in range(num_clients):
            cnt = counts[i]
            if cnt > 0:
                client_idx[i].extend(c_idx[ptr:ptr+cnt].tolist())
                ptr += cnt
    return client_idx

# -------------------------
# Aggregator
# -------------------------
class Aggregator:
    def __init__(self, strategy: str = "sample_weight"):
        assert strategy in ("sample_weight", "uniform")
        self.strategy = strategy

    def aggregate(self, global_state: Dict[str, torch.Tensor], client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        # all aggregation on CPU tensors
        K = len(client_states)
        if self.strategy == "sample_weight":
            total = float(sum(client_sizes))
            weights = [s / total for s in client_sizes]
        else:
            weights = [1.0 / K] * K

        new_state = {}
        for k in global_state.keys():
            agg = torch.zeros_like(global_state[k], dtype=torch.float32)
            for cs, w in zip(client_states, weights):
                agg += cs[k].float() * w
            new_state[k] = agg
        return new_state

# -------------------------
# Client and Server classes
# -------------------------
@dataclass
class ClientResult:
    state: Dict[str, torch.Tensor]
    num_samples: int

class Client:
    def __init__(self, client_id: int, loader: DataLoader, device: torch.device):
        self.cid = client_id
        self.loader = loader
        self.device = device

    def local_train(self, model_fn, global_state: Dict[str, torch.Tensor], epochs: int, lr: float, weight_decay: float = 0.0) -> ClientResult:
        model = model_fn()
        model.load_state_dict(global_state)
        model.to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        total = 0
        for epoch in range(epochs):
            loop = tqdm(self.loader, desc=f"Client {self.cid} epoch {epoch+1}", leave=False)
            running = 0.0
            count = 0
            for x, y in loop:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running += loss.item() * x.size(0)
                count += x.size(0)
            total += count
        state = to_cpu_state(model.state_dict())
        return ClientResult(state=state, num_samples=total)

class Server:
    def __init__(self, model_fn, aggregator: Aggregator, device: torch.device):
        self.model_fn = model_fn
        self.aggregator = aggregator
        self.device = device
        # initialize global model on CPU, weights on CPU
        m = model_fn()
        self.global_state = to_cpu_state(m.state_dict())

    def distribute_and_collect(self, clients: List[Client], rounds_cfg: dict) -> Tuple[float, Dict]:
        """Run one full federated training loop with provided clients.
           rounds_cfg contains keys: rounds, local_epochs, lr, weight_decay, clients_per_round, log_csv.
        """
        rounds = rounds_cfg["rounds"]
        local_epochs = rounds_cfg["local_epochs"]
        lr = rounds_cfg["lr"]
        weight_decay = rounds_cfg.get("weight_decay", 0.0)
        clients_per_round = rounds_cfg.get("clients_per_round", len(clients))
        csv_path = rounds_cfg.get("log_csv")

        history = []
        best_acc = 0.0
        global_model = self.model_fn()
        for r in range(1, rounds + 1):
            LOG.info(f"Starting round {r}/{rounds}")
            selected = list(range(len(clients)))
            if clients_per_round < len(clients):
                selected = random.sample(range(len(clients)), clients_per_round)

            client_states = []
            client_sizes = []
            for idx in selected:
                client = clients[idx]
                res = client.local_train(self.model_fn, self.global_state, local_epochs, lr, weight_decay)
                client_states.append(res.state)
                client_sizes.append(res.num_samples)

            # aggregate on CPU
            new_state = self.aggregator.aggregate(self.global_state, client_states, client_sizes)
            self.global_state = {k: v.clone() for k, v in new_state.items()}

            # evaluate on server-side central validation/test if provided
            global_model.load_state_dict(self.global_state)
            metrics = rounds_cfg.get("eval_fn")(global_model) if rounds_cfg.get("eval_fn") else {}
            LOG.info(f"Round {r} metrics: {metrics}")
            history.append({"round": r, **metrics})

            # save csv incrementally
            if csv_path:
                write_csv_row(csv_path, {"round": r, **metrics})

            # optional model saving
            if metrics.get("accuracy", 0) > best_acc:
                best_acc = metrics.get("accuracy", 0)
                if rounds_cfg.get("save_path"):
                    torch.save(self.global_state, rounds_cfg["save_path"])
                    LOG.info(f"Saved best model to {rounds_cfg['save_path']}")
        return best_acc, {"history": history}

def write_csv_row(path: str, row: dict):
    file_exists = Path(path).exists()
    with open(path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# -------------------------
# Evaluation wrapper
# -------------------------
def make_eval_fn(model_fn, eval_loader: Optional[DataLoader], device: torch.device):
    def eval_fn(model: nn.Module) -> Dict[str, float]:
        if eval_loader is None:
            return {}
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        criterion = nn.CrossEntropyLoss(reduction="sum")
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_sum += criterion(out, y).item()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return {"loss": loss_sum / total, "accuracy": correct / total}
    return eval_fn

# -------------------------
# Main orchestration
# -------------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    LOG.info(f"Using device: {device}")

    train_transform, eval_transform = get_transforms(img_size=args.img_size, to_3ch=args.to_3ch)
    train_ds, val_ds, test_ds = load_imagefolder(Path(args.data_root), train_transform, eval_transform)
    LOG.info(f"Train size: {len(train_ds)}; val: {len(val_ds) if val_ds else 'N/A'}; test: {len(test_ds) if test_ds else 'N/A'}")

    # Partition: custom unbalanced split (example proportions)
    # For 4 clients with different dataset sizes, you can pass `proportions=[0.4,0.3,0.2,0.1]`
    if args.partition == "iid":
        partitions = iid_partition(train_ds, args.num_clients, proportions=args.client_proportions)
    else:
        partitions = dirichlet_partition(train_ds, args.num_clients, alpha=args.dir_alpha)

    client_loaders = []
    client_sizes = []
    for cid in range(args.num_clients):
        idxs = partitions[cid]
        subset = Subset(train_ds, idxs)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        client_loaders.append(loader)
        client_sizes.append(len(idxs))
        LOG.info(f"Client {cid} samples: {len(idxs)}")

    # create client objects
    model_fn = lambda: (get_resnet18(num_classes=len(train_ds.classes), pretrained=False, in_channels=3)
                        if args.model == "resnet18"
                        else get_simple_cnn(num_classes=len(train_ds.classes), in_channels=3))
    clients = [Client(cid, client_loaders[cid], device) for cid in range(args.num_clients)]

    # evaluation loader wrapper
    eval_loader = None
    if args.eval_on == "val" and val_ds:
        eval_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.eval_on == "test" and test_ds:
        eval_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    eval_fn = make_eval_fn(model_fn, eval_loader, device)

    aggregator = Aggregator(strategy=args.aggregation)
    server = Server(model_fn, aggregator, device)

    rounds_cfg = {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clients_per_round": args.clients_per_round,
        "eval_fn": eval_fn,
        "save_path": args.save_path,
        "log_csv": args.log_csv,
    }

    best_acc, info = server.distribute_and_collect(clients, rounds_cfg)
    LOG.info(f"Best acc achieved: {best_acc}")
    if args.final_eval and test_ds:
        # final test
        global_model = model_fn()
        global_model.load_state_dict(server.global_state)
        final_eval = eval_fn(global_model)
        LOG.info(f"Final test metrics: {final_eval}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--client_proportions", type=json.loads, default=None,
                        help='JSON list of proportions for IID split e.g. "[0.4,0.3,0.2,0.1]"')
    parser.add_argument("--partition", type=str, choices=["iid","dirichlet"], default="iid")
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--aggregation", type=str, choices=["sample_weight","uniform"], default="sample_weight")
    parser.add_argument("--model", type=str, choices=["simple","resnet18"], default="simple")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--to_3ch", action="store_true", help="convert grayscale to 3 channels")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--clients_per_round", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_path", type=str, default="best_global.pth")
    parser.add_argument("--log_csv", type=str, default="fed_history.csv")
    parser.add_argument("--eval_on", type=str, choices=["val","test","none"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final_eval", action="store_true", help="run final eval on test set if available")
    args = parser.parse_args()
    main(args)
