import numpy as np
import torch
from torch.utils.data import Dataset

class PianoTranscriptionDataset(Dataset):
    def __init__(self, data_dir, split, augment=False, max_samples=None):
        log_mel_file_name = split + '_log_mel_segments.npy'
        piano_roll_file_name = split + '_piano_roll_segments.npy'

        # Load precomputed data
        self.log_mel_segments = np.load(data_dir / log_mel_file_name)
        self.piano_roll_segments = np.load(data_dir / piano_roll_file_name)

        # Tranpose [n_segments, n_mels, n_frames] -> [n_segments, n_frames, n_mels]
        self.log_mel_segments = self.log_mel_segments.transpose(0, 2, 1)

        # Trim samples, if specified
        if max_samples:
            self.log_mel_segments = self.log_mel_segments[:max_samples, :, :]
            self.piano_roll_segments = self.piano_roll_segments[:max_samples, :]

        # Convert to float tensors
        self.log_mel_segments = torch.tensor(self.log_mel_segments, dtype=torch.float32).unsqueeze(1)
        self.piano_roll_segments = torch.tensor(self.piano_roll_segments, dtype=torch.float32)

    def __len__(self):
        return len(self.log_mel_segments)
    
    def __getitem__(self, idx):
        x = self.log_mel_segments[idx]
        y = self.piano_roll_segments[idx]

        return x, y