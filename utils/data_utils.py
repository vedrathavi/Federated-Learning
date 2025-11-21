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


def load_datasets(data_dir):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    valset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    return trainset, valset, testset


def split_clients(dataset, num_clients=4, seed=42):
    """Split dataset into client datasets with different sizes.
    
    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of clients (fixed)
        seed: Random seed for reproducibility
        
    Returns:
        List of client datasets
    """
    total_len = len(dataset)

    # Default proportions for 4 clients (heterogeneous split)
    # Client 1: 10%, Client 2: 25%, Client 3: 35%, Client 4: 30%
    if num_clients == 4:
        proportions = [0.1, 0.25, 0.35, 0.3]
    else:
        # Equal split for other numbers of clients
        proportions = [1.0 / num_clients] * num_clients
    
    sizes = [int(total_len * p) for p in proportions]

    # Handle rounding errors
    diff = total_len - sum(sizes)
    sizes[-1] += diff

    generator = torch.Generator().manual_seed(seed)
    client_datasets = random_split(dataset, sizes, generator=generator)

    return client_datasets


def get_client_loaders(client_datasets, batch_size=16, num_workers=0):
    return [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for ds in client_datasets
    ]


def get_test_loader(testset, batch_size=32, num_workers=0):
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
