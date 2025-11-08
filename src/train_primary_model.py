import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from data_pipeline.dataset import PianoTranscriptionDataset
from models.primary import PianoTranscriptionModel

def train(model, train_data, val_data=None, batch_size=16, learning_rate=0.01, num_epochs=10, save_path=None, save_checkpoint=True):
    torch.manual_seed(42)
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss(pos_weight=train_data.pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0

        for log_mel, frame_targets in train_loader:
            log_mel = log_mel.to(device)
            frame_targets = frame_targets.to(device)

            optimizer.zero_grad()
            outputs = model(log_mel) # Output: (batch_size, n_frames, 88)

            loss = criterion(outputs, frame_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for log_mel, frame_targets in val_loader:
                log_mel = log_mel.to(device)
                frame_targets = frame_targets.to(device)

                outputs = model(log_mel) # Output: (batch_size, n_frames, 88)
                loss = criterion(outputs, frame_targets)

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses[epoch] = avg_train_loss
        val_losses[epoch] = avg_val_loss

        # Save checkpoints
        if save_checkpoint and save_path:
            torch.save(model.state_dict(), f'AA.pth') # TODO: Create function to obtain model name (use for checkpoint file name)

        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_val_loss:.4f} | Validation Loss: {avg_train_loss:.4f}')

    # if save_path:
        np.savetxt(save_path / 'pri_train_loss.csv', train_losses)
        np.savetxt(save_path / 'pri_val_loss.csv', val_losses)

if __name__ == '__main__':
    model = PianoTranscriptionModel()

    train_dataset = PianoTranscriptionDataset(config.PRIMARY_TRAIN_DIR, 'train')
    val_dataset = PianoTranscriptionDataset(config.PRIMARY_VAL_DIR, 'val')
    test_dataset = PianoTranscriptionDataset(config.PRIMARY_TEST_DIR, 'test')

    train(model, train_dataset, train_dataset, num_epochs=30, save_path=pathlib.Path('.'), save_checkpoint=True)
