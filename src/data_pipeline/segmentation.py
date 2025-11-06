import numpy as np

def segment_pair_primary(
    log_mel: np.ndarray,
    piano_roll: np.ndarray,
    sr: float,
    hop_length: int = 512,
    segment_duration: float = 5.0,
    drop_last: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a log-Mel spectrogram and piano roll pair into equal time chunks.

    This function is used to segment data for the primary model. Each segment
    of the log-Mel spectrogram corresponds directly to a segment of the piano
    roll of the same length, preserving a 1:1 alignment between input and label.

    Parameters
    ----------
    log_mel : np.ndarray
        A 2D NumPy array of shape (n_mels, n_frames) containing the log-Mel
        spectrogram (in decibels).
    piano_roll : np.ndarray
        A 2D NumPy array of shape (128, n_frames) representing the piano roll.
    sr : float
        The sampling rate of the audio in Hz.
    hop_length : int, default=512
        The number of samples between successive frames.
    segment_duration : float, default=5.0
        The duration of each segment in seconds.
    drop_last : bool, default=True
        Indicates whether or not to discard leftover frames that do not fit a
        full segment.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - np.ndarray
            An array of all log-Mel spectrogram segments.
        - np.ndarray
            An array of all piano roll segments.
    """
    log_mel_segments = []
    piano_roll_segments = []

    frame_duration = hop_length / sr
    segment_frames = int(segment_duration / frame_duration)

    # Should be equal, but take minimum as a precaution
    total_frames = min(log_mel.shape[1], piano_roll.shape[1])
    num_full_segments = total_frames // segment_frames
    remainder = total_frames % segment_frames

    for i in range(num_full_segments):
        start = i * segment_frames
        end = start + segment_frames

        log_mel_segments.append(log_mel[:, start:end])
        piano_roll_segments.append(piano_roll[:, start:end])

    # Handle remainder frames at the end, if necessary
    if remainder > 0 and not drop_last:
        start = num_full_segments * segment_frames
        end = total_frames

        # Pad to full segment length (use zeros)
        log_mel_tail = log_mel[:, start:end]
        piano_roll_tail = piano_roll[:, start:end]

        pad_width = segment_frames - log_mel_tail.shape[1]
        log_mel_tail = np.pad(log_mel_tail, ((0, 0), (0, pad_width)))
        piano_roll_tail = np.pad(piano_roll_tail, ((0, 0), (0, pad_width)))

        log_mel_segments.append(log_mel_tail)
        piano_roll_segments.append(piano_roll_tail)

    return np.stack(log_mel_segments), np.stack(piano_roll_segments)
