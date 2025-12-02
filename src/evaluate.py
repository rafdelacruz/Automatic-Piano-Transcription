import torch
import torch.nn as nn

import config
from data_pipeline.dataset import PianoTranscriptionDataset
from models.baseline import AllConv
from metrics import classification_metrics, confusion_matrix_metrics
from torch.utils.data import DataLoader

def evaluate(model: nn.Module, criterion: nn.Module, data_loader: DataLoader, threshold: float = 0.5) -> None:
    num_samples = 0
    loss = 0.0
    tp, fp, fn = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for log_mel, frame_target in data_loader:
            logits = model(log_mel) # Output: (batch_size, 88)
            loss = criterion(logits, frame_target)

            prediction = (torch.sigmoid(logits) >= threshold).float()
            tp += confusion_matrix_metrics.count_true_positives(prediction, frame_target)
            fp += confusion_matrix_metrics.count_false_positives(prediction, frame_target)
            fn += confusion_matrix_metrics.count_false_negatives(prediction, frame_target)

            loss += loss.item() * frame_target.size(0)
            num_samples += frame_target.size(0)

    avg_loss = loss / num_samples
    f1 = classification_metrics.get_f1_from_counts(tp, fp, fn)

    print(f'Loss: {avg_loss:.4f} | F1 Score: {f1:.4f}')

def evaluate_baseline_model(experiment_name: str, checkpoint_num: int, data: PianoTranscriptionDataset, threshold: float):
    model = AllConv()
    state = torch.load(config.EXPERIMENTS_DIR  / experiment_name / 'checkpoints' / f'epoch_{checkpoint_num}.pth')
    model.load_state_dict(state)

    data_loader = DataLoader(data)
    criterion = nn.BCEWithLogitsLoss()

    evaluate(model, criterion, data_loader, threshold=threshold)

if __name__ == '__main__':
    experiment_name = 'baseline_model'
    checkpoint_num = 50
    data_dir = config.BASELINE_TEST_DIR
    threshold = 0.9

    dataset = PianoTranscriptionDataset(
        log_mel_segments_file=(data_dir / 'test_log_mel_segments.npy'),
        piano_roll_segments_file=(data_dir / 'test_piano_roll_segments.npy'),
        split='test',
        max_samples=2500
    )

    evaluate_baseline_model(
        experiment_name=experiment_name,
        checkpoint_num=checkpoint_num,
        data=dataset,
        threshold=threshold
    )
