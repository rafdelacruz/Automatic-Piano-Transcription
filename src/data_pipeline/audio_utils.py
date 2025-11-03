import pathlib

import librosa
import numpy as np

def load_audio(filepath: pathlib.Path, target_sr: float = 16000) -> np.ndarray:
    """
    Load an audio file and resample it to the target sampling rate.

    Parameters
    ----------
    filepath : pathlib.Path
        A Path object pointing to the audio file.
    target_sr : float, default=16000
        The target sampling rate to resample the audio to (in Hz).

    Returns
    -------
    np.ndarray
        A NumPy array containing the audio waveform as a time series at the
        specified sampling rate.
    """
    audio, _ = librosa.load(filepath, sr=target_sr)
    return audio

def compute_log_mel_spectrogram(
    audio: np.ndarray, sr: float, n_mels: int = 229, hop_length: int = 512
) -> np.ndarray:
    """
    Compute the log-Mel spectrogram of an audio waveform.

    Parameters
    ----------
    audio : np.ndarray
        A NumPy array containing the audio waveform.
    sr : float
        The sampling rate of the audio in Hz.
    n_mels : int, default=229
        The number of Mel bands to generate.
    hop_length : int, default=512
        The number of samples between successive frames.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_mels, n_frames) containing the log-Mel
        spectrogram (in decibels).
    """
    mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel
