import numpy as np

from data_pipeline.dataset import PianoTranscriptionDataset

# Assume input: [n_segments, 88, n_frames]
def count_note_activations(piano_roll: np.ndarray) -> np.ndarray:
    return piano_roll.sum(axis=(0, 2)).astype(np.int32)

# Assume piano roll tensors: [n_segments, n_frames, 88]
def count_note_activations_in_dataset(dataset: PianoTranscriptionDataset) -> np.ndarray:
    activations = np.zeros(88, dtype=np.int32)

    for _, piano_roll in dataset:
        piano_roll = piano_roll.numpy()
        activations += piano_roll.sum(axis=(0)).astype(np.int32)

    return activations