# utils.py
import torch

def get_device():
    """
    Determines the device to use for tensor operations (GPU or CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """
    Adjusts edge indices for batch processing in graph neural networks.
    
    Parameters:
    - org_edge_index (torch.Tensor): Original edge indices (2, edge_num).
    - batch_num (int): Number of batches.
    - node_num (int): Number of nodes per batch.

    Returns:
    - torch.Tensor: Adjusted edge indices for the batches.
    """
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()
