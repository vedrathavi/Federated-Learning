"""
Adaptive FedAvg (IID, 4 clients) for Pneumonia detection using a compact CNN.
Windows/CPU-safe defaults with detailed logging and artifacts.

Current setup:
    - 4-layer CNN backbone instead of the previous pretrained setup
    - PERF_TEMPERATURE = 5.0 for stronger adaptive weighting
    - NUM_ROUNDS = 20
    - LOCAL_EPOCHS = 5
    - Richer train augmentation: RandomRotation(10) + ColorJitter

By default, outputs are written to:
    adaptive_fedavg/outputs/
You can override dataset/output locations via CLI flags.
"""

import argparse
import os
import copy
import json
import math
import random
import shutil
from datetime import datetime

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image, ImageFile

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support

ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# -----------------------
# Config / Hyperparams
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "nih-dataset")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEFAULT_OUTPUT_HISTORY_DIR = os.path.join(BASE_DIR, "outputs_history")

DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
OUTPUT_HISTORY_DIR = DEFAULT_OUTPUT_HISTORY_DIR

NUM_CLIENTS = 4
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
LOCAL_BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 128
NUM_WORKERS = 0
SEED = 42

# Adaptive FedAvg controls
BETA_SIZE = 0.6          # contribution of client data-size weight
BETA_PERF = 0.4          # contribution of client performance weight
PERF_TEMPERATURE = 5.0   # raised from 2.0 — sharpens adaptive weight toward better-performing clients
MIN_CLIENT_WEIGHT = 1e-6

# Optional robust loss for heavy class imbalance.
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Decision threshold for binary prediction at evaluation time.
EVAL_THRESHOLD = 0.2
# Reduce class reweighting intensity to avoid over-compensation.
POS_WEIGHT_SCALE = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive FedAvg training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Dataset root containing train/ and test/ folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write current run outputs",
    )
    parser.add_argument(
        "--output-history-dir",
        type=str,
        default=DEFAULT_OUTPUT_HISTORY_DIR,
        help="Directory to archive previous outputs",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip auto-archiving of existing output-dir contents",
    )
    return parser.parse_args()


def configure_paths(data_dir, output_dir, output_history_dir):
    global DATA_DIR, OUTPUT_DIR, OUTPUT_HISTORY_DIR, PLOTS_DIR
    DATA_DIR = os.path.abspath(data_dir)
    OUTPUT_DIR = os.path.abspath(output_dir)
    OUTPUT_HISTORY_DIR = os.path.abspath(output_history_dir)
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_HISTORY_DIR, exist_ok=True)


def archive_previous_outputs():
    if not os.path.isdir(OUTPUT_DIR):
        return

    existing_items = os.listdir(OUTPUT_DIR)
    if len(existing_items) == 0:
        return

    archive_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(OUTPUT_HISTORY_DIR, f"run_{archive_stamp}")
    os.makedirs(archive_dir, exist_ok=True)

    for item_name in existing_items:
        src = os.path.join(OUTPUT_DIR, item_name)
        dst = os.path.join(archive_dir, item_name)
        shutil.move(src, dst)

    print(f"Archived previous outputs to: {archive_dir}")


def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def filter_valid_samples(dataset):
    valid_samples = []
    for sample_path, label in dataset.samples:
        if is_valid_image(sample_path):
            valid_samples.append((sample_path, label))

    filtered_count = len(dataset.samples) - len(valid_samples)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} unreadable images from {dataset.root}")

    dataset.samples = valid_samples
    dataset.targets = [label for _, label in valid_samples]
    return dataset


class NIHPneumoniaBinaryDataset(Dataset):
    """Binary NIH dataset where label 1 means Pneumonia is present, else 0."""

    def __init__(self, samples, transform, root):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform
        self.root = root
        self.classes = ["non_pneumonia", "pneumonia"]
        self.class_to_idx = {"non_pneumonia": 0, "pneumonia": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def infer_dataset_name(data_dir):
    name = os.path.basename(os.path.normpath(data_dir))
    return name if name else "dataset"


def make_dataset_tag(dataset_name):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in dataset_name.lower())
    cleaned = cleaned.strip("_")
    return cleaned if cleaned else "dataset"


def resolve_nih_images_root(data_dir):
    candidates = [
        os.path.join(data_dir, "images-224", "images-224"),
        os.path.join(data_dir, "images-224"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def build_nih_binary_splits(data_dir, train_transform, eval_transform):
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    train_list_path = os.path.join(data_dir, "train_val_list_NIH.txt")
    test_list_path = os.path.join(data_dir, "test_list_NIH.txt")
    images_root = resolve_nih_images_root(data_dir)

    if images_root is None:
        raise FileNotFoundError(f"NIH images folder not found under {data_dir}")

    metadata = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])
    labels_by_image = dict(zip(metadata["Image Index"], metadata["Finding Labels"]))

    def label_from_findings(findings):
        labels = [item.strip() for item in str(findings).split("|") if item.strip()]
        return 1 if "Pneumonia" in labels else 0

    def read_split(split_list_path):
        with open(split_list_path, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]

        split_samples = []
        missing_images = 0
        missing_labels = 0
        for image_name in names:
            findings = labels_by_image.get(image_name)
            if findings is None:
                missing_labels += 1
                continue
            image_path = os.path.join(images_root, image_name)
            if not os.path.isfile(image_path):
                missing_images += 1
                continue
            split_samples.append((image_path, label_from_findings(findings)))

        return split_samples, missing_images, missing_labels

    train_samples, train_missing_images, train_missing_labels = read_split(train_list_path)
    test_samples, test_missing_images, test_missing_labels = read_split(test_list_path)

    if len(train_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("NIH split construction produced an empty train or test split")

    print(
        "NIH split summary:",
        f"train={len(train_samples)} (missing_images={train_missing_images}, missing_labels={train_missing_labels}),",
        f"test={len(test_samples)} (missing_images={test_missing_images}, missing_labels={test_missing_labels})",
    )

    train_pos = sum(label for _, label in train_samples)
    test_pos = sum(label for _, label in test_samples)
    print(
        "NIH class balance:",
        f"train pneumonia={train_pos}, non_pneumonia={len(train_samples) - train_pos};",
        f"test pneumonia={test_pos}, non_pneumonia={len(test_samples) - test_pos}",
    )

    train_dataset = NIHPneumoniaBinaryDataset(train_samples, transform=train_transform, root=data_dir)
    test_dataset = NIHPneumoniaBinaryDataset(test_samples, transform=eval_transform, root=data_dir)

    return train_dataset, test_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Model / Train / Eval
# -----------------------
class PneumoniaCNN(nn.Module):
    """Compact CNN with 4 convolutional layers for binary pneumonia classification."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def create_model():
    return PneumoniaCNN(in_channels=3)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = self.alpha * ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()


def local_train(model, dataloader, device, epochs=1, lr=1e-3, weight_decay=1e-4):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_dataset = dataloader.dataset
    if hasattr(train_dataset, "targets"):
        labels = [int(label) for label in train_dataset.targets]
    else:
        labels = [int(label) for _, label in train_dataset]

    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    # avoid division by zero
    pos_weight = torch.tensor([POS_WEIGHT_SCALE * (num_neg / (num_pos + 1e-6))], dtype=torch.float32, device=device)

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    running_loss = 0.0
    total_batches = 0

    epoch_bar = tqdm(range(epochs), desc="Local epochs", leave=False, dynamic_ncols=True)

    for _ in epoch_bar:
        epoch_loss = 0.0
        epoch_batches = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_loss = float(loss.item())
            running_loss += batch_loss
            total_batches += 1
            epoch_loss += batch_loss
            epoch_batches += 1

        scheduler.step()
        epoch_bar.set_postfix(avg_loss=f"{(epoch_loss / max(epoch_batches, 1)):.4f}")

    avg_loss = running_loss / max(total_batches, 1)
    return model.state_dict(), avg_loss


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    ys = []
    probs = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        ps = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
        probs.extend(ps.tolist())
        ys.extend(labels.numpy().tolist())

    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs >= EVAL_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        ys,
        preds,
        average="binary",
        zero_division=0,
    )
    specificity = tn / max(tn + fp, 1)
    sensitivity = recall
    balanced_accuracy = 0.5 * (recall + specificity)

    try:
        auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    acc = accuracy_score(ys, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "balanced_accuracy": float(balanced_accuracy),
        "y_true": ys,
        "y_prob": probs,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def iid_split(num_samples, num_clients):
    indices = list(range(num_samples))
    random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [list(s) for s in splits]


def get_loader_from_indices(dataset, indices, batch_size, train_transform, eval_transform, shuffle_train=False):
    ds = copy.copy(dataset)
    ds.samples = [dataset.samples[i] for i in indices]
    ds.targets = [dataset.targets[i] for i in indices]
    ds.transform = train_transform if shuffle_train else eval_transform

    if shuffle_train:
        labels = [int(label) for label in ds.targets]
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos

        class_counts = [max(num_neg, 1), max(num_pos, 1)]
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = [float(class_weights[label]) for label in labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
        return loader

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


def adaptive_fedavg(local_weights, local_sizes, local_performances, beta_size=0.6, beta_perf=0.4, temperature=2.0):
    total_samples = sum(local_sizes)

    size_weights = np.array([size / max(total_samples, 1) for size in local_sizes], dtype=np.float64)

    perf = np.array(local_performances, dtype=np.float64)
    perf = np.clip(perf, 1e-6, 1.0)

    perf_scaled = np.power(perf, temperature)
    perf_weights = perf_scaled / np.sum(perf_scaled)

    adaptive_weights = beta_size * size_weights + beta_perf * perf_weights
    adaptive_weights = np.maximum(adaptive_weights, MIN_CLIENT_WEIGHT)
    adaptive_weights = adaptive_weights / np.sum(adaptive_weights)

    new_global = {}
    for key in local_weights[0].keys():
        if local_weights[0][key].dtype == torch.float32:
            new_global[key] = torch.zeros_like(local_weights[0][key])
        else:
            new_global[key] = local_weights[0][key].clone()

    for client_weight, lw in zip(adaptive_weights, local_weights):
        for k in lw.keys():
            if lw[k].dtype == torch.float32:
                new_global[k] += lw[k] * float(client_weight)

    return new_global, size_weights.tolist(), perf_weights.tolist(), adaptive_weights.tolist()


def compute_weight_drift(global_prev, global_new):
    sq_sum = 0.0
    numel = 0
    for k in global_prev.keys():
        if global_prev[k].dtype == torch.float32:
            diff = (global_new[k].cpu() - global_prev[k].cpu()).float()
            sq_sum += float(torch.sum(diff * diff).item())
            numel += diff.numel()
    if numel == 0:
        return 0.0
    return float(math.sqrt(sq_sum / numel))


def plot_roc_curve(y_true, y_prob, save_path, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Final Global Model ({dataset_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(tn, fp, fn, tp, save_path, title):
    matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def run(args):
    configure_paths(args.data_dir, args.output_dir, args.output_history_dir)

    if not args.no_archive:
        archive_previous_outputs()
    ensure_dirs()
    set_seed(SEED)

    if NUM_CLIENTS != 4:
        raise ValueError("This implementation is fixed to IID with exactly 4 clients.")

    print("Using device:", DEVICE)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data directory: {DATA_DIR}")

    dataset_name = infer_dataset_name(DATA_DIR)
    dataset_tag = make_dataset_tag(dataset_name)
    print(f"Dataset name: {dataset_name}")

    def plot_path(stem):
        return os.path.join(PLOTS_DIR, f"{dataset_tag}_{stem}.png")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),                          # new: small rotation robustness
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # new: lighting variation robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    nih_metadata = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
    if os.path.isfile(nih_metadata):
        full_train, test_dataset = build_nih_binary_splits(DATA_DIR, train_transform, eval_transform)
    else:
        train_root = os.path.join(DATA_DIR, "train")
        test_root = os.path.join(DATA_DIR, "test")

        if not os.path.isdir(train_root):
            raise FileNotFoundError(f"Train folder not found at {train_root}")
        if not os.path.isdir(test_root):
            raise FileNotFoundError(f"Test folder not found at {test_root}")

        full_train = datasets.ImageFolder(train_root, transform=train_transform)
        test_dataset = datasets.ImageFolder(test_root, transform=eval_transform)

        full_train = filter_valid_samples(full_train)
        test_dataset = filter_valid_samples(test_dataset)

    print("Classes (train):", full_train.class_to_idx)

    client_splits = iid_split(len(full_train), NUM_CLIENTS)
    clients = []
    for split in client_splits:
        random.shuffle(split)
        cutoff = int(0.8 * len(split))
        train_idxs = split[:cutoff]
        val_idxs = split[cutoff:]
        clients.append({"train_idxs": train_idxs, "val_idxs": val_idxs})

    print("Client dataset sizes (train,val):", [(len(c["train_idxs"]), len(c["val_idxs"])) for c in clients])

    test_loader = DataLoader(test_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    global_model = create_model().to(DEVICE)
    global_weights = copy.deepcopy(global_model.state_dict())

    global_round_rows = []
    client_round_rows = []
    weight_drift_rows = []

    client_test_acc_history = {cid: [] for cid in range(NUM_CLIENTS)}
    global_acc_history = []
    global_f1_history = []
    global_auc_history = []

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n=== Adaptive FedAvg Round {rnd}/{NUM_ROUNDS} ===")
        selected_clients = list(range(NUM_CLIENTS))
        print("Selected clients:", selected_clients)

        for cid in range(NUM_CLIENTS):
            client_test_acc_history[cid].append(np.nan)

        local_weights = []
        local_sizes = []
        local_perf_f1 = []

        prev_global_cpu = {k: v.detach().cpu().clone() for k, v in global_weights.items()}

        for cid in selected_clients:
            train_idxs = clients[cid]["train_idxs"]
            if len(train_idxs) == 0:
                continue

            train_loader = get_loader_from_indices(
                dataset=full_train,
                indices=train_idxs,
                batch_size=LOCAL_BATCH_SIZE,
                train_transform=train_transform,
                eval_transform=eval_transform,
                shuffle_train=True,
            )

            local_model = create_model().to(DEVICE)
            local_model.load_state_dict(global_weights)

            updated_weights, train_loss = local_train(
                local_model,
                train_loader,
                DEVICE,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )

            val_loader = get_loader_from_indices(
                dataset=full_train,
                indices=clients[cid]["val_idxs"],
                batch_size=LOCAL_BATCH_SIZE,
                train_transform=train_transform,
                eval_transform=eval_transform,
                shuffle_train=False,
            )

            test_metrics_local = evaluate_model(local_model, val_loader, DEVICE)
            local_acc = test_metrics_local["accuracy"]
            local_precision = test_metrics_local["precision"]
            local_recall = test_metrics_local["recall"]
            local_f1 = test_metrics_local["f1"]
            local_auc = test_metrics_local["auc"]
            local_specificity = test_metrics_local["specificity"]
            local_sensitivity = test_metrics_local["sensitivity"]
            local_balanced_accuracy = test_metrics_local["balanced_accuracy"]

            local_weights.append({k: v.detach().cpu().clone() for k, v in updated_weights.items()})
            local_sizes.append(len(train_idxs))
            local_perf_f1.append(local_f1)

            client_test_acc_history[cid][-1] = local_acc

            client_round_rows.append({
                "round": rnd,
                "client": cid,
                "n_train_samples": len(train_idxs),
                "local_train_loss": train_loss,
                "local_test_accuracy": local_acc,
                "local_test_precision": local_precision,
                "local_test_recall": local_recall,
                "local_test_f1": local_f1,
                "local_test_auc": local_auc,
                "local_test_specificity": local_specificity,
                "local_test_sensitivity": local_sensitivity,
                "local_test_balanced_accuracy": local_balanced_accuracy,
                "size_weight": np.nan,
                "performance_weight": np.nan,
                "adaptive_weight": np.nan,
            })

            print(
                f"  Client {cid} -> TrainLoss: {train_loss:.4f}, "
                f"Acc: {local_acc:.4f}, Prec: {local_precision:.4f}, Rec: {local_recall:.4f}, "
                f"F1: {local_f1:.4f}, AUC: {local_auc:.4f}, Spec: {local_specificity:.4f}, "
                f"Sens: {local_sensitivity:.4f}, BalAcc: {local_balanced_accuracy:.4f}"
            )

        if len(local_weights) == 0:
            print("No local updates in this round.")
            continue

        new_global_cpu, size_w, perf_w, adapt_w = adaptive_fedavg(
            local_weights=local_weights,
            local_sizes=local_sizes,
            local_performances=local_perf_f1,
            beta_size=BETA_SIZE,
            beta_perf=BETA_PERF,
            temperature=PERF_TEMPERATURE,
        )

        selected_client_rows = [r for r in client_round_rows if r["round"] == rnd]
        for idx, row in enumerate(selected_client_rows):
            row["size_weight"] = size_w[idx]
            row["performance_weight"] = perf_w[idx]
            row["adaptive_weight"] = adapt_w[idx]

        global_weights = {k: v.to(DEVICE) for k, v in new_global_cpu.items()}
        global_model.load_state_dict(global_weights)

        drift_l2 = compute_weight_drift(prev_global_cpu, new_global_cpu)
        weight_drift_rows.append({"round": rnd, "global_weight_drift_l2": drift_l2})

        global_metrics = evaluate_model(global_model, test_loader, DEVICE)
        global_acc_history.append(global_metrics["accuracy"])
        global_f1_history.append(global_metrics["f1"])
        global_auc_history.append(global_metrics["auc"])

        global_round_rows.append({
            "round": rnd,
            "global_accuracy": global_metrics["accuracy"],
            "global_precision": global_metrics["precision"],
            "global_recall": global_metrics["recall"],
            "global_f1": global_metrics["f1"],
            "global_auc": global_metrics["auc"],
            "global_specificity": global_metrics["specificity"],
            "global_sensitivity": global_metrics["sensitivity"],
            "global_balanced_accuracy": global_metrics["balanced_accuracy"],
            "global_tn": global_metrics["tn"],
            "global_fp": global_metrics["fp"],
            "global_fn": global_metrics["fn"],
            "global_tp": global_metrics["tp"],
            "mean_client_accuracy": float(np.nanmean([r["local_test_accuracy"] for r in selected_client_rows])),
            "std_client_accuracy": float(np.nanstd([r["local_test_accuracy"] for r in selected_client_rows])),
            "global_weight_drift_l2": drift_l2,
        })

        weights_msg = ", ".join([f"C{idx}:{w:.4f}" for idx, w in enumerate(adapt_w)])
        print(
            f"Global -> Acc: {global_metrics['accuracy']:.4f}, Prec: {global_metrics['precision']:.4f}, "
            f"Rec: {global_metrics['recall']:.4f}, F1: {global_metrics['f1']:.4f}, AUC: {global_metrics['auc']:.4f}, "
            f"Spec: {global_metrics['specificity']:.4f}, Sens: {global_metrics['sensitivity']:.4f}, "
            f"BalAcc: {global_metrics['balanced_accuracy']:.4f}"
            f" | Drift(L2): {drift_l2:.6f}"
        )
        print(f"Adaptive weights: {weights_msg}")

    final_metrics = {
        "accuracy": float(global_acc_history[-1]) if global_acc_history else float("nan"),
        "f1": float(global_f1_history[-1]) if global_f1_history else float("nan"),
        "auc": float(global_auc_history[-1]) if global_auc_history else float("nan"),
    }

    print("\n=== Final Global Model Metrics ===")
    print(final_metrics)

    # Per-client final evaluation on local validation splits
    per_client_results = []
    for cid, client_info in enumerate(clients):
        val_idxs = client_info["val_idxs"]
        if len(val_idxs) == 0:
            per_client_results.append({
                "client": cid,
                "n_samples": 0,
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
                "specificity": float("nan"),
                "sensitivity": float("nan"),
                "balanced_accuracy": float("nan"),
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tp": np.nan,
            })
            continue

        val_loader = get_loader_from_indices(
            dataset=full_train,
            indices=val_idxs,
            batch_size=LOCAL_BATCH_SIZE,
            train_transform=train_transform,
            eval_transform=eval_transform,
            shuffle_train=False,
        )
        m = evaluate_model(global_model, val_loader, DEVICE)
        per_client_results.append({
            "client": cid,
            "n_samples": len(val_idxs),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
            "specificity": m["specificity"],
            "sensitivity": m["sensitivity"],
            "balanced_accuracy": m["balanced_accuracy"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
        })

    # Final global ROC on shared test set
    final_test_eval = evaluate_model(global_model, test_loader, DEVICE)
    if len(np.unique(final_test_eval["y_true"])) > 1:
        plot_roc_curve(
            final_test_eval["y_true"],
            final_test_eval["y_prob"],
            save_path=plot_path("roc_curve_global"),
            dataset_name=dataset_name,
        )

    plot_confusion_matrix(
        final_test_eval["tn"],
        final_test_eval["fp"],
        final_test_eval["fn"],
        final_test_eval["tp"],
        save_path=plot_path("confusion_matrix_global"),
        title=f"Global Model Confusion Matrix ({dataset_name})",
    )

    # Save tabular outputs
    global_df = pd.DataFrame(global_round_rows)
    client_round_df = pd.DataFrame(client_round_rows)
    drift_df = pd.DataFrame(weight_drift_rows)
    per_client_df = pd.DataFrame(per_client_results)

    global_df.to_csv(os.path.join(OUTPUT_DIR, "global_round_metrics.csv"), index=False)
    client_round_df.to_csv(os.path.join(OUTPUT_DIR, "client_round_metrics.csv"), index=False)
    drift_df.to_csv(os.path.join(OUTPUT_DIR, "weight_drift.csv"), index=False)
    per_client_df.to_csv(os.path.join(OUTPUT_DIR, "per_client_results.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "final_global_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(str(final_metrics))
    with open(os.path.join(OUTPUT_DIR, "final_global_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_dir": DATA_DIR,
            "dataset_name": dataset_name,
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
            "device": str(DEVICE),
            "adaptive": {
                "beta_size": BETA_SIZE,
                "beta_perf": BETA_PERF,
                "perf_temperature": PERF_TEMPERATURE,
            },
            "iid_split": True,
            "fixed_clients": 4,
            "model": "PneumoniaCNN (4 conv layers + batchnorm + adaptive avg pool)",
            "augmentation": "flip + rotation10 + colorjitter",
        },
        "final_metrics": final_metrics,
        "best_global_accuracy": float(np.nanmax(global_acc_history)) if global_acc_history else float("nan"),
        "best_global_f1": float(np.nanmax(global_f1_history)) if global_f1_history else float("nan"),
        "best_global_auc": float(np.nanmax(global_auc_history)) if global_auc_history else float("nan"),
    }

    with open(os.path.join(OUTPUT_DIR, "adaptive_fedavg_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -----------------------
    # Plots
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(global_acc_history) + 1), global_acc_history, marker="o", linewidth=2, label="Global Accuracy")
    plt.plot(range(1, len(global_f1_history) + 1), global_f1_history, marker="s", linewidth=2, label="Global F1")
    plt.plot(range(1, len(global_auc_history) + 1), global_auc_history, marker="^", linewidth=2, label="Global AUC")
    plt.plot(range(1, len(global_round_rows) + 1), [row["global_balanced_accuracy"] for row in global_round_rows], marker="d", linewidth=2, label="Global Balanced Acc")
    plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
    plt.ylabel("Metric Value", fontsize=12, fontweight="bold")
    plt.title(f"Global Metrics vs Rounds (Adaptive FedAvg) - {dataset_name}", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("global_metrics_vs_rounds"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(weight_drift_rows) + 1), [row["global_weight_drift_l2"] for row in weight_drift_rows], marker="o", linewidth=2, color="tab:purple")
    plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
    plt.ylabel("L2 Drift", fontsize=12, fontweight="bold")
    plt.title(f"Global Weight Drift vs Rounds - {dataset_name}", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path("weight_drift_vs_rounds"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLIENTS))
    for cid in range(NUM_CLIENTS):
        rounds = list(range(1, len(client_test_acc_history[cid]) + 1))
        accs = client_test_acc_history[cid]

        smoothed = []
        ema = None
        for acc in accs:
            if not np.isnan(acc):
                if ema is None:
                    ema = acc
                else:
                    ema = 0.3 * acc + 0.7 * ema
                smoothed.append(ema)
            else:
                smoothed.append(np.nan)

        plt.plot(rounds, smoothed, marker="o", linewidth=2.5, markersize=6, color=colors[cid], label=f"Client {cid}")

    plt.xlabel("Federated Round", fontsize=13, fontweight="bold")
    plt.ylabel("Shared Test Accuracy", fontsize=13, fontweight="bold")
    plt.title(f"Client-wise Accuracy Trajectories (Smoothed) - {dataset_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10, ncol=2)
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.xticks(range(1, NUM_ROUNDS + 1))
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(plot_path("client_accuracy_over_rounds"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cid in range(NUM_CLIENTS):
        client_series = [row["local_test_accuracy"] for row in client_round_rows if row["client"] == cid]
        plt.plot(range(1, len(client_series) + 1), client_series, linestyle="--", alpha=0.6, label=f"Client {cid} local")

    plt.plot(range(1, len(global_acc_history) + 1), global_acc_history, color="black", linewidth=3, label="Global aggregated")
    plt.xlabel("Round", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title(f"Client Convergence Under Adaptive Aggregation - {dataset_name}", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path("client_convergence_effect"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    client_df = pd.DataFrame(per_client_results)
    plt.bar(client_df["client"].astype(str), client_df["accuracy"], label="Accuracy", alpha=0.85)
    plt.plot(client_df["client"].astype(str), client_df["f1"], marker="o", linewidth=2, label="F1")
    plt.plot(client_df["client"].astype(str), client_df["auc"], marker="s", linewidth=2, label="AUC")
    plt.xlabel("Client ID", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.title(f"Final Per-Client Validation Metrics - {dataset_name}", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("final_metrics_per_client"), dpi=150)
    plt.close()

    print("\nTraining complete. Detailed outputs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run(parse_args())
