# Training loop for AllConv model
# Based on: Kelz et al., "On the Potential of Simple Framewise Approaches to Piano Transcription"
# https://arxiv.org/pdf/1612.05153

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.baseline import AllConv

def train(model, train_data, val_data=None, batch_size=16, num_epochs=10, save_path=None):
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)

    # Halve learning rate every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0

        for log_mel, frame_target in train_loader:
            log_mel = log_mel.to(device)
            frame_target = frame_target.to(device)

            optimizer.zero_grad()
            outputs = model(log_mel) # Output: (batch_size, 88)

            loss = criterion(outputs, frame_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for log_mel, frame_target in val_loader:
                log_mel = log_mel.to(device)
                frame_target = frame_target.to(device)

                outputs = model(log_mel) # Output: (batch_size, 88)
                loss = criterion(outputs, frame_target)

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        # Save checkpoint
        if save_path:
            torch.save(model.state_dict(), 'state.pth')

        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')


if __name__ == '__main__':
    model = AllConv()
    