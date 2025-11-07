import torch

def get_precision(predicted_notes, true_notes):
    tp = torch.sum((predicted_notes == 1) & (true_notes == 1)).item()
    fp = torch.sum((predicted_notes == 1) & (true_notes == 0)).item()

    return 0 if tp + fp == 0 else tp / (tp + fp)

def get_recall(predicted_notes, true_notes):
    tp = torch.sum((predicted_notes == 1) & (true_notes == 1)).item()
    fn = torch.sum((predicted_notes == 0) & (true_notes == 1)).item()

    return 0 if tp + fn == 0 else tp / (tp + fn)

def get_f1(predicted_notes, true_notes):
    prec = get_precision(predicted_notes, true_notes)
    rec = get_recall(predicted_notes, true_notes)

    return 0 if prec + rec == 0 else 2 * (prec * rec) / (prec + rec)