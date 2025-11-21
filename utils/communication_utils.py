"""
Utilities for tracking communication overhead in federated learning.
Tracks bytes sent and received per communication round.
"""

import torch
import sys
from typing import Dict, List


def get_model_size_bytes(model: torch.nn.Module) -> int:
    """Calculate the size of model parameters in bytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in bytes
    """
    total_bytes = 0
    for param in model.parameters():
        # Each parameter's size in bytes = num_elements * bytes_per_element
        total_bytes += param.numel() * param.element_size()
    return total_bytes


def get_state_dict_size_bytes(state_dict: Dict) -> int:
    """Calculate the size of a state dictionary in bytes.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Size in bytes
    """
    total_bytes = 0
    for key, tensor in state_dict.items():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


def calculate_communication_cost(num_clients: int, model_size_bytes: int) -> Dict[str, int]:
    """Calculate communication costs for one round of federated learning.
    
    In standard FedAvg:
    - Server sends global model to each client: num_clients * model_size
    - Each client sends updated model back to server: num_clients * model_size
    
    Args:
        num_clients: Number of participating clients in the round
        model_size_bytes: Size of the model in bytes
        
    Returns:
        Dictionary with 'bytes_sent_to_clients', 'bytes_received_from_clients', 'total_bytes'
    """
    bytes_sent_to_clients = num_clients * model_size_bytes
    bytes_received_from_clients = num_clients * model_size_bytes
    total_bytes = bytes_sent_to_clients + bytes_received_from_clients
    
    return {
        'bytes_sent_to_clients': bytes_sent_to_clients,
        'bytes_received_from_clients': bytes_received_from_clients,
        'total_bytes': total_bytes,
        'model_size_bytes': model_size_bytes,
        'num_clients': num_clients
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable format (KB, MB, GB).
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


class CommunicationTracker:
    """Track communication costs across multiple rounds."""
    
    def __init__(self):
        self.round_costs = []
        self.total_bytes = 0
        
    def add_round(self, round_num: int, num_clients: int, model_size_bytes: int):
        """Add communication cost for a round.
        
        Args:
            round_num: Round number
            num_clients: Number of clients in this round
            model_size_bytes: Size of model in bytes
        """
        cost = calculate_communication_cost(num_clients, model_size_bytes)
        cost['round'] = round_num
        self.round_costs.append(cost)
        self.total_bytes += cost['total_bytes']
        
    def get_summary(self) -> Dict:
        """Get summary statistics of communication costs.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.round_costs:
            return {}
            
        return {
            'total_rounds': len(self.round_costs),
            'total_bytes': self.total_bytes,
            'total_bytes_formatted': format_bytes(self.total_bytes),
            'avg_bytes_per_round': self.total_bytes / len(self.round_costs),
            'avg_bytes_per_round_formatted': format_bytes(self.total_bytes / len(self.round_costs)),
            'model_size_bytes': self.round_costs[0]['model_size_bytes'] if self.round_costs else 0,
            'model_size_formatted': format_bytes(self.round_costs[0]['model_size_bytes']) if self.round_costs else '0 B'
        }
    
    def get_round_costs(self) -> List[Dict]:
        """Get list of all round costs."""
        return self.round_costs
