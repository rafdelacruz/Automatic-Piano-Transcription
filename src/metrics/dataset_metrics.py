import numpy as np

# Assume input: [n_segments, 88, n_frames]
def count_note_activations(piano_roll: np.ndarray) -> np.ndarray:
    return piano_roll.sum(axis=(0, 2)).astype(np.int32)