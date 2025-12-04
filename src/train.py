import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from data_pipeline.dataset import PianoTranscriptionDataset
from models.baseline import AllConv
from models.primary import PianoTranscriptionModel
from metrics import classification_metrics, confusion_matrix_metrics

def grid_search_threshold(
    experiment_name: str,
    num_epochs: int,
    data_loader: DataLoader,
    min_threshold: float,
    max_threshold: float,
    step: float,
    primary: bool = False,
    print_progress: bool = True
) -> tuple[int, float, float]:
    num_points = int(round((max_threshold - min_threshold) / step)) + 1
    thresholds = np.linspace(min_threshold, max_threshold, num_points)
    checkpoint_dir = config.EXPERIMENTS_DIR / experiment_name / 'checkpoints'

    best_checkpoint = 0
    best_threshold = 0.0
    best_f1 = 0.0

    model = PianoTranscriptionModel() if primary else AllConv()

    for epoch in range(num_epochs):
        print(f'SEARCHING CHECKPOINT {epoch + 1}')
        state = torch.load(checkpoint_dir / f'epoch_{epoch + 1}.pth')
        model.load_state_dict(state)
        model.eval()
        
        for threshold in thresholds:
            tp, fp, fn = 0, 0, 0

            with torch.no_grad():
                for log_mel, frame_target in data_loader:
                    logits = model(log_mel)
                    prediction = (torch.sigmoid(logits) >= threshold).float()

                    tp += confusion_matrix_metrics.count_true_positives(prediction, frame_target)
                    fp += confusion_matrix_metrics.count_false_positives(prediction, frame_target)
                    fn += confusion_matrix_metrics.count_false_negatives(prediction, frame_target)

            f1 = classification_metrics.get_f1_from_counts(tp, fp, fn)
            print(f'Threshold {threshold:.2f}: {(f1 * 100):.2f}%')

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_checkpoint = epoch + 1
        
        print()

    return best_checkpoint, best_threshold, best_f1

def get_training_config_str(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    num_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> str:
    config_str = []

    config_str.append('===== Training Configurations =====')
    config_str.append(f'Model: {model.__class__.__name__}')
    config_str.append(f'Loss Function: {criterion.__class__.__name__}')
    config_str.append(f'Optimizer: {optimizer.__class__.__name__}')
    config_str.append(f'Optimizer Parameters: {optimizer.defaults}')
    config_str.append(f'Scheduler: {scheduler.__class__.__name__ if scheduler else None}')
    config_str.append(f'Number of Epochs: {num_epochs}')
    config_str.append(f'Learning Rate: {optimizer.param_groups[0]['lr'] if scheduler is None else optimizer.param_groups[0]['initial_lr']}')
    config_str.append(f'Batch Size: {train_loader.batch_size}')
    config_str.append(f'Train Dataset Size: {len(train_loader.dataset)}')
    config_str.append(f'Validation Dataset Size: {len(val_loader.dataset)}')
    config_str.append('===================================')

    return '\n'.join(config_str)

def train(
    model: nn.Module,
    train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    num_epochs: int = 10,
    save_path: pathlib.Path | None = None,
    save_checkpoint: bool = True
) -> None:
    torch.manual_seed(42)

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    train_f1_scores = np.zeros(num_epochs)
    val_f1_scores = np.zeros(num_epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if hasattr(criterion, 'pos_weight') and criterion.pos_weight is not None:
        criterion.pos_weight = criterion.pos_weight.to(device)

    for epoch in range(num_epochs):
         # ---------- Training ----------
        model.train()

        train_loss = 0.0
        num_train_samples = 0
        train_tp = 0
        train_fp = 0
        train_fn = 0

        for log_mel, frame_target in train_loader:
            log_mel = log_mel.to(device)
            frame_target = frame_target.to(device)

            optimizer.zero_grad()
            logits = model(log_mel) # Output: (batch_size, 88)

            loss = criterion(logits, frame_target)
            loss.backward()
            optimizer.step()

            prediction = (torch.sigmoid(logits) >= 0.5).float()
            train_tp += confusion_matrix_metrics.count_true_positives(prediction, frame_target)
            train_fp += confusion_matrix_metrics.count_false_positives(prediction, frame_target)
            train_fn += confusion_matrix_metrics.count_false_negatives(prediction, frame_target)

            train_loss += loss.item() * frame_target.size(0)
            num_train_samples += frame_target.size(0)

        # ---------- Validation ----------
        model.eval()

        val_loss = 0.0
        num_val_samples = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0

        with torch.no_grad():
            for log_mel, frame_target in val_loader:
                log_mel = log_mel.to(device)
                frame_target = frame_target.to(device)

                logits = model(log_mel) # Output: (batch_size, 88)
                loss = criterion(logits, frame_target)

                prediction = (torch.sigmoid(logits) >= 0.5).float()
                val_tp += confusion_matrix_metrics.count_true_positives(prediction, frame_target)
                val_fp += confusion_matrix_metrics.count_false_positives(prediction, frame_target)
                val_fn += confusion_matrix_metrics.count_false_negatives(prediction, frame_target)

                val_loss += loss.item() * frame_target.size(0)
                num_val_samples += frame_target.size(0)

        avg_train_loss = train_loss / num_train_samples
        avg_val_loss = val_loss / num_val_samples
        train_f1 = classification_metrics.get_f1_from_counts(train_tp, train_fp, train_fn)
        val_f1 = classification_metrics.get_f1_from_counts(val_tp, val_fp, val_fn)

        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss
        train_f1_scores[epoch] = train_f1
        val_f1_scores[epoch] = val_f1

        if scheduler is not None:
            scheduler.step()

        # Save checkpoints
        if save_checkpoint and save_path:
            torch.save(model.state_dict(), save_path / 'checkpoints' / f'epoch_{epoch + 1}.pth')

        width = len(str(num_epochs))
        print(
            f'Epoch {(epoch + 1):>{width}}/{num_epochs}: '
            f'Train Loss = {train_losses[epoch]:.4f} | '
            f'Train F1 = {train_f1_scores[epoch]:.4f} | '
            f'Validation Loss = {val_losses[epoch]:.4f} | '
            f'Validation F1 = {val_f1_scores[epoch]:.4f}'
        )

    if save_path:
        np.savetxt(save_path / 'train_loss.csv', train_losses)
        np.savetxt(save_path / 'val_loss.csv', val_losses)
        np.savetxt(save_path / 'train_f1_scores.csv', train_f1_scores)
        np.savetxt(save_path / 'val_f1_scores.csv', val_f1_scores)

        training_config = get_training_config_str(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader
        )

        with open(save_path / 'training_config.txt', 'w') as f:
            f.write(training_config)

def train_baseline_model(
    experiment_name: str,
    train_data: PianoTranscriptionDataset,
    val_data: PianoTranscriptionDataset,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1.0,
    momentum: float = 0.9,
    save_checkpoint: bool = True
) -> None:
    model = AllConv()

    # Create experiment directory
    experiment_dir = config.EXPERIMENTS_DIR / experiment_name
    experiment_checkpoints_dir = experiment_dir / 'checkpoints'
    experiment_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss(pos_weight=train_data.pos_weight)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_path=experiment_dir,
        save_checkpoint=save_checkpoint
    )

def train_primary_model(
    experiment_name: str,
    train_data: PianoTranscriptionDataset,
    val_data: PianoTranscriptionDataset,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    save_checkpoint: bool = True
) -> None:
    model = PianoTranscriptionModel()

    # Create experiment directory
    experiment_dir = config.EXPERIMENTS_DIR / experiment_name
    experiment_checkpoints_dir = experiment_dir / 'checkpoints'
    experiment_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss(pos_weight=train_data.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=experiment_dir,
        save_checkpoint=save_checkpoint
    )

if __name__ == '__main__':
    experiment_name = 'baseline_model'
    train_dir = config.BASELINE_TRAIN_DIR
    val_dir = config.BASELINE_VAL_DIR

    train_dataset = PianoTranscriptionDataset(
        log_mel_segments_file=(train_dir / 'log_mel_segments.npy'),
        piano_roll_segments_file=(train_dir / 'piano_roll_segments.npy'),
        split='train'
    )

    val_dataset = PianoTranscriptionDataset(
        log_mel_segments_file=(val_dir / 'log_mel_segments.npy'),
        piano_roll_segments_file=(val_dir / 'piano_roll_segments.npy'),
        split='train'
    )

    train_baseline_model(
        experiment_name=experiment_name,
        train_data=train_dataset,
        val_data=val_dataset,
        num_epochs=50
    )