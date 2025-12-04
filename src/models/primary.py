import torch
import torch.nn as nn
import torch.nn.functional as F

class PianoTranscriptionModel(nn.Module):
    def __init__(self):
        super(PianoTranscriptionModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=(1, 1))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(32)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2)) # Reduces frequency only

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)
        self.drop3 = nn.Dropout2d(0.1)
        self.drop4 = nn.Dropout2d(0.1)
        self.drop5 = nn.Dropout(0.25)

        self.rnn = nn.LSTM(
            input_size=(229 // 4) * 32,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(128 * 2, 88)

    def forward(self, x):
        # x: (batch, 1, frames, mels)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.max_pool(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.drop3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.drop4(x)

        x = x.permute(0, 2, 1, 3) # Output: (batch, frames, channels, mel)
        x = x.flatten(start_dim=2) # Output: (batch, frames, features)

        x, _ = self.rnn(x)
        x = self.drop5(x)
        x = self.fc(x)

        return x
