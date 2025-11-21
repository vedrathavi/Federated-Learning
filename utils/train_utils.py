import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm


def train_local(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                device: torch.device,
                val_loader: Optional[torch.utils.data.DataLoader] = None,
                epochs: int = 1,
                lr: float = 1e-3,
                print_per_epoch: bool = True,
                client_id: Optional[int] = None) -> Tuple[Dict, List[dict]]:
    """Train a local copy of the model and report per-epoch metrics.

    Returns (state_dict, history) where history is a list of dicts with keys:
    'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    # Create a fresh local model instance and load the global weights.
    # This avoids deepcopy issues when the global model is on CUDA and
    # prevents accidental sharing of GPU tensors between copies.
    local_model = model.__class__()
    local_model.load_state_dict(deepcopy(model.state_dict()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=lr)
    local_model.to(device)

    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        # Training
        local_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_desc = f"Client {client_id} Epoch {epoch} - train" if client_id is not None else f"Epoch {epoch} - train"
        pbar = tqdm(train_loader, desc=train_desc, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # update running postfix so user sees live metrics without printing lines
            if total > 0:
                pbar.set_postfix(train_loss=f"{running_loss/total:.4f}", train_acc=f"{correct/total:.4f}")
        pbar.close()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        # Validation (optional)
        val_loss = None
        val_acc = None
        if val_loader is not None:
            local_model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                val_desc = f"Client {client_id} Epoch {epoch} - val" if client_id is not None else f"Epoch {epoch} - val"
                vpbar = tqdm(val_loader, desc=val_desc, leave=False)
                for images, labels in vpbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item() * labels.size(0)
                    preds = outputs.argmax(dim=1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)
                    if v_total > 0:
                        vpbar.set_postfix(val_loss=f"{v_loss/v_total:.4f}", val_acc=f"{v_correct/v_total:.4f}")
                vpbar.close()

            if v_total > 0:
                val_loss = v_loss / v_total
                val_acc = v_correct / v_total
            else:
                val_loss = 0.0
                val_acc = 0.0

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        if print_per_epoch:
            prefix = f"Client {client_id} - " if client_id is not None else ""
            # use tqdm.write to avoid corrupting active progress bars
            if val_loader is not None:
                tqdm.write(prefix + f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                tqdm.write(prefix + f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    return local_model.state_dict(), history
