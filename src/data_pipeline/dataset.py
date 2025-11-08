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

        # Transpose [n_segments, 88, n_frames] -> [n_segments, n_frames, 88]
        if self.piano_roll_segments.ndim == 3:
            self.piano_roll_segments = self.piano_roll_segments.transpose((0, 2, 1))

        # Trim samples, if specified
        if max_samples:
            self.log_mel_segments = self.log_mel_segments[:max_samples, :, :]
            self.piano_roll_segments = self.piano_roll_segments[:max_samples, :]

        # Compute positive weight if class is a training dataset
        self.pos_weight = -1

        if split == 'train':
            num_pos = self.piano_roll_segments.sum(axis=0)
            num_neg = self.piano_roll_segments.shape[0] - num_pos

            self.pos_weight = num_neg / (num_pos + 1e-6)
            self.pos_weight = np.clip(self.pos_weight, 1, 20)
            self.pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32)

        # Convert to float tensors
        self.log_mel_segments = torch.tensor(self.log_mel_segments, dtype=torch.float32).unsqueeze(1)
        self.piano_roll_segments = torch.tensor(self.piano_roll_segments, dtype=torch.float32)

    def __len__(self):
        return len(self.log_mel_segments)
    
    def __getitem__(self, idx):
        x = self.log_mel_segments[idx]
        y = self.piano_roll_segments[idx]

        return x, y