import os
import random
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

def load_datasets(data_dir):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    valset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    return dataset, valset, testset

def split_clients(dataset, num_clients=4):
    total_len = len(dataset)

    # Uneven proportions for 4 hospitals
    proportions = [0.1, 0.25, 0.35, 0.3]
    sizes = [int(total_len * p) for p in proportions]

    # Fix rounding issue
    diff = total_len - sum(sizes)
    sizes[-1] += diff  # adjust the last one so total matches exactly

    datasets_split = random_split(dataset, sizes)
    return datasets_split

def get_client_loaders(client_datasets, batch_size=16, num_workers: int = 0):
    """Return DataLoaders for each client. Explicitly set num_workers to avoid
    spawning worker processes that can re-import the main module on Windows.
    """
    return [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for ds in client_datasets]
