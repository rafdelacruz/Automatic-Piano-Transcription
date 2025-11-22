import librosa
import numpy as np

def add_noise_to_log_mel(log_mel: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """
    Add random Gaussian noise to a log-Mel spectrogram.

    Parameters
    ----------
    log_mel : np.ndarray
        A log-Mel spectrogram of shape (n_mels, n_frames) to add noise to.
    noise_std : float, default=0.05
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    np.ndarray
        A copy of the input log-Mel spectrogram with added Gaussian noise.
    """
    noisy_log_mel = log_mel.copy()

    noise = np.random.randn(*log_mel.shape) * noise_std
    noisy_log_mel += noise

    return noisy_log_mel

def pitch_shift(
    audio: np.ndarray, piano_roll: np.ndarray, sr: float, shift_val: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pitch-shift both an audio waveform and its corresponding piano roll.

    Parameters
    ----------
    audio : np.ndarray
        A 1D NumPy array containing the audio waveform as a time series at the
        specified sampling rate.
    piano_roll : np.ndarray
        A 2D NumPy array of shape (128, n_frames) representing the piano roll.
    sr : float
        The sampling rate of the audio in Hz.
    shift_val : int
        The number of semitones to shift by (positive values shift up, negative
        shift down).
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - np.ndarray
            The pitch-shifted audio waveform.
        - np.ndarray
            The pitch-shifted piano roll.
    """
    # Shift audio
    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift_val)

    # Shift piano roll
    shifted_piano_roll = np.zeros_like(piano_roll)

    if shift_val > 0:
        shifted_piano_roll[shift_val:, :] = piano_roll[:-shift_val, :]
    elif shift_val < 0:
        shifted_piano_roll[:-abs(shift_val), :] = piano_roll[abs(shift_val):, :]
    else:
        shifted_piano_roll[:] = piano_roll

    return shifted_audio, shifted_piano_roll

def get_safe_pitch_shift_range(piano_roll: np.ndarray) -> tuple[int, int]:
    """
    Compute the safe pitch shift range for a given piano roll.

    This function calculates the maximum upward and downward pitch shift that
    can be applied without moving any active notes out of the valid piano MIDI
    range (0-87).

    Parameters
    ----------
    piano_roll : np.ndarray
        A 2D NumPy array of shape (128, n_frames) representing the piano roll.

    Returns
    -------
    tuple[int, int]
        A tuple containing:
        - int
            The maximum upward pitch shift.
        - int
            The maximum downward pitch shift.    
    """
    active_rows = np.where(piano_roll.any(axis=1))[0]

    max_up = 87 - active_rows.max()
    max_down = -active_rows.min()

    return max_up, max_down