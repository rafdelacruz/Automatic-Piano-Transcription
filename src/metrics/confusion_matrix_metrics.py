import torch

def count_true_positives(predicted_notes: torch.Tensor, true_notes: torch.Tensor) -> int:
    return ((predicted_notes == 1) & (true_notes == 1)).sum().item()

def count_false_positives(predicted_notes: torch.Tensor, true_notes: torch.Tensor) -> int:
    return ((predicted_notes == 1) & (true_notes == 0)).sum().item()

def count_false_negatives(predicted_notes: torch.Tensor, true_notes: torch.Tensor) -> int:
    return ((predicted_notes == 0) & (true_notes == 1)).sum().item()