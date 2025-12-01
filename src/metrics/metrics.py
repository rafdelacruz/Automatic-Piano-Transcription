import torch

def get_precision(predicted_notes: torch.Tensor, true_notes: torch.Tensor) -> float:
    """
    Compute the precision score for binary note predictions.

    Precision is defined as: precision = TP / (TP + FP), where TP is true
    positives and FP is false positives.

    Parameters
    ----------
    predicted_notes : torch.Tensor
        Binary tensor of predicted note activations, where 1 indicates an
        active note and 0 indicates silence.
    true_notes : torch.Tensor
        Binary tensor of ground-truth note activations with the same shape
        as predicted_notes.

    Returns
    -------
    float
        The precision score value.
    """
    tp = torch.sum((predicted_notes == 1) & (true_notes == 1)).item()
    fp = torch.sum((predicted_notes == 1) & (true_notes == 0)).item()

    return 0.0 if tp + fp == 0 else tp / (tp + fp)

def get_recall(predicted_notes: torch.Tensor, true_notes: torch.Tensor) -> float:
    """
    Compute the recall score for binary note predictions.

    Recall is defined as: recall = TP / (TP + FN), where TP is true
    positives and FN is false negatives.

    Parameters
    ----------
    predicted_notes : torch.Tensor
        Binary tensor of predicted note activations, where 1 indicates an
        active note and 0 indicates silence.
    true_notes : torch.Tensor
        Binary tensor of ground-truth note activations with the same shape
        as predicted_notes.

    Returns
    -------
    float
        The recall score value.
    """
    tp = torch.sum((predicted_notes == 1) & (true_notes == 1)).item()
    fn = torch.sum((predicted_notes == 0) & (true_notes == 1)).item()

    return 0.0 if tp + fn == 0 else tp / (tp + fn)

def get_f1(predicted_notes: torch.Tensor, true_notes: torch.Tensor) -> float:
    """
    Compute the F1 score for binary note predictions.

    The F1 score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    predicted_notes : torch.Tensor
        Binary tensor of predicted note activations, where 1 indicates an
        active note and 0 indicates silence.
    true_notes : torch.Tensor
        Binary tensor of ground-truth note activations with the same shape
        as predicted_notes.

    Returns
    -------
    float
        The F1 score value.
    """
    prec = get_precision(predicted_notes, true_notes)
    rec = get_recall(predicted_notes, true_notes)

    return 0.0 if prec + rec == 0 else 2 * (prec * rec) / (prec + rec)

def get_precision_from_counts(true_positives: int, false_positives: int) -> float:
    return 0.0 if true_positives + false_positives == 0 else true_positives / (true_positives + false_positives)

def get_recall_from_counts(true_positives: int, false_negatives: int) -> float:
    return 0.0 if true_positives + false_negatives == 0 else true_positives / (true_positives + false_negatives)

def get_f1_from_counts(
    true_positives: int, false_positives: int, false_negatives: int
) -> float:
    prec = get_precision_from_counts(true_positives, false_positives)
    rec = get_recall_from_counts(true_positives, false_negatives)
    return 0.0 if prec + rec == 0 else 2 * (prec * rec) / (prec + rec)
