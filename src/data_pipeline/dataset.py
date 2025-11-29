import numpy as np
import pathlib
import torch
from torch.utils.data import Dataset

class PianoTranscriptionDataset(Dataset):
    """
    PyTorch Dataset for loading precomputed log-mel spectrogram segments and
    corresponding piano-roll labels for piano transcription.

    Parameters
    ----------
    log_mel_segments_file : pathlib.Path
        A Path object pointing to the file that holds the log-Mel segments.
    piano_roll_segments_file : pathlib.Path
        A Path object pointing to the file that holds the piano roll segments.
    split : {'train', 'val', 'test'}
        A string representing which dataset split to load.
    normalize : bool, default=True
        Indicates whether to standard normalize the log-Mel spectrograms.
    augment : bool, default=False
        Indicates whether or to apply data augentations.
    max_samples : int, default=None
        If provided, limits the number of samples loaded.

    Attributes
    ----------
    log_mel_segments : torch.Tensor
        Tensor of shape (n_segments, 1, n_frames, n_mels) containing the
        log_mel spectrograms.
    piano_roll_segments : torch.Tensor
        Tensor of shape (n_segments, n_frames, 88) for the primary model data
        or shape (n_segments, 88) for the baseline model data, containing the
        piano_roll labels.
    pos_weight : torch.Tensor
        Per-note positive class weight for handling imbalance (only for
        training set).

    """
    def __init__(
        self,
        log_mel_segments_file: pathlib.Path,
        piano_roll_segments_file: pathlib.Path,
        split: str,
        normalize: bool = True,
        augment: bool = False,
        max_samples: int = None
    ) -> None:
        # Load precomputed data
        self.log_mel_segments = np.load(log_mel_segments_file)
        self.piano_roll_segments = np.load(piano_roll_segments_file)

        # Standard normalize log-Mel spectograms
        if normalize:
            mean = np.mean(self.log_mel_segments)
            std = np.std(self.log_mel_segments)

            self.log_mel_segments = (self.log_mel_segments - mean) / (std + 1e-6)

        # Tranpose [n_segments, n_mels, n_frames] -> [n_segments, n_frames, n_mels]
        self.log_mel_segments = self.log_mel_segments.transpose(0, 2, 1)

        # Transpose [n_segments, 88, n_frames] -> [n_segments, n_frames, 88]
        self.piano_roll_segments = self.piano_roll_segments.transpose((0, 2, 1))

        # Trim samples, if specified
        if max_samples:
            np.random.seed(42)

            perm = np.random.permutation(len(self.log_mel_segments))
            self.log_mel_segments = self.log_mel_segments[perm]
            self.piano_roll_segments = self.piano_roll_segments[perm]

            self.log_mel_segments = self.log_mel_segments[:max_samples, :, :]
            self.piano_roll_segments = self.piano_roll_segments[:max_samples, :, :]

        # Compute positive weight if class is a training dataset
        self.pos_weight = -1

        if split == 'train':
            num_pos = self.piano_roll_segments.sum(axis=(0, 1))
            num_neg = self.piano_roll_segments.shape[0] * self.piano_roll_segments.shape[1] - num_pos

            self.pos_weight = num_neg / (num_pos + 1e-6)
            self.pos_weight = np.clip(self.pos_weight, 1, 20)
            self.pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32)

        # Convert to float tensors
        self.log_mel_segments = torch.tensor(self.log_mel_segments, dtype=torch.float32).unsqueeze(1)
        self.piano_roll_segments = torch.tensor(self.piano_roll_segments, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Total number of (log-mel, piano-roll) pairs.
        """
        return len(self.log_mel_segments)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample pair.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - torch.Tensor
                Tensor of log-mel spectrograms.
            - torch.Tensor
                Tensor of piano-roll labels.
        """
        x = self.log_mel_segments[idx]
        y = self.piano_roll_segments[idx]

        return x, y