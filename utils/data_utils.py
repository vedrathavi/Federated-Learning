import os
import random
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(data_dir, img_size: int = 224, to_3ch: bool = False):
    """Load ImageFolder datasets with configurable transforms.

    Args:
        data_dir: root dataset directory containing train/val/test
        img_size: resize target
        to_3ch: convert grayscale to 3 channels when True (for pretrained ResNet)

    Returns:
        trainset, valset, testset
    """
    train_transforms = []
    eval_transforms = []

    if to_3ch:
        train_transforms.append(transforms.Grayscale(num_output_channels=3))
        eval_transforms.append(transforms.Grayscale(num_output_channels=3))
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
    else:
        train_transforms.append(transforms.Grayscale())
        eval_transforms.append(transforms.Grayscale())
        normalize_mean = [0.5]
        normalize_std = [0.5]

    train_transforms += [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ]

    eval_transforms += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ]

    train_tf = transforms.Compose(train_transforms)
    eval_tf = transforms.Compose(eval_transforms)

    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    valset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=eval_tf)
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=eval_tf)

    return trainset, valset, testset


def split_clients(dataset, num_clients=4, seed=42, partition: str = 'dirichlet', alpha: float = 0.5):
    """Split dataset into client datasets.

    Supports IID equal splitting and Dirichlet non-IID splitting.

    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of clients
        seed: RNG seed
        partition: 'iid' or 'dirichlet'
        alpha: concentration parameter for Dirichlet (smaller => more heterogeneous)

    Returns:
        list of Subset datasets (one per client)
    """
    total_len = len(dataset)
    # labels for each sample (ImageFolder stores (path, class_idx) in samples)
    labels = np.array([dataset.samples[i][1] for i in range(total_len)])

    if partition == 'iid':
        proportions = [1.0 / num_clients] * num_clients
        sizes = [int(total_len * p) for p in proportions]
        diff = total_len - sum(sizes)
        sizes[-1] += diff
        generator = torch.Generator().manual_seed(seed)
        client_datasets = random_split(dataset, sizes, generator=generator)
        return client_datasets

    # Dirichlet non-iid partition
    np.random.seed(seed)
    parts = {i: [] for i in range(num_clients)}
    classes = np.unique(labels)
    for c in classes:
        c_idx = np.where(labels == c)[0]
        np.random.shuffle(c_idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(c_idx)).astype(int)
        # fix rounding
        while counts.sum() < len(c_idx):
            counts[np.argmax(proportions)] += 1
        ptr = 0
        for i in range(num_clients):
            cnt = counts[i]
            if cnt > 0:
                parts[i].extend(c_idx[ptr:ptr+cnt].tolist())
                ptr += cnt

    from torch.utils.data import Subset
    client_datasets = [Subset(dataset, parts[i]) for i in range(num_clients)]
    return client_datasets


def get_client_loaders(client_datasets, batch_size=16, num_workers=0):
    return [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for ds in client_datasets
    ]


def get_test_loader(testset, batch_size=32, num_workers=0):
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
