import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(audio: np.ndarray, sr: float) -> None:
    """
    Plot the waveform of an audio file.

    Parameters
    ----------
    audio : np.ndarray
        A NumPy array containing the audio waveform as a time series.
    sr : float
        The sampling rate of the audio waveform.
    """
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.show()

def plot_log_mel_spectrogram(
    log_mel: np.ndarray, sr: float, hop_length: int
) -> None:
    """
    Plot a log-Mel spectrogram.

    Parameters
    ----------
    log_mel : np.ndarray
        A NumPy array of shape (n_mels, n_frames) containing the log-Mel
        spectrogram (in decibels).
    sr : float
        The sampling rate of the audio in Hz.
    hop_length : int
        The number of samples between successive frames.
    """
    librosa.display.specshow(
        log_mel, sr=sr,
        hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap='magma'
    )
    plt.title('Log-Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')

    plt.show()

def plot_piano_roll(piano_roll: np.ndarray) -> None:
    """
    Plot a piano roll (multi or single frame).

    Parameters
    ----------
    piano_roll : np.ndarray
        A NumPy array of shape (88) or (88, n_frames) representing the piano roll.
    """
    if piano_roll.ndim == 1: # One frame
        plt.bar(np.arange(88), piano_roll, color='purple')
        plt.title('Piano Roll')
        plt.yticks([0, 1])
        plt.xlabel('Piano Key Number')
        plt.ylabel('Activation')
    else: # Multiple frames
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='magma_r')
        plt.title('Piano Roll')
        plt.xlabel('Time Frame')
        plt.ylabel('Piano Key Number')

    plt.show()

def plot_log_mel_and_piano_roll(
    log_mel: np.ndarray, piano_roll: np.ndarray, sr: float, hop_length: int
) -> None:
    """
    Plot a log-Mel spectrogram and its corresponding piano roll.

    Parameters
    ----------
    log_mel : np.ndarray
        A NumPy array of shape (n_mels, n_frames) containing the log-Mel
        spectrogram (in decibels).
    piano_roll : np.ndarray
        A NumPy array of shape (88) or (88, n_frames) representing the piano roll.
    sr : float
        The sampling rate of the audio in Hz.
    hop_length : int
        The number of samples between successive frames.
    
    """
    plt.subplot(1, 2, 1)
    librosa.display.specshow(
        log_mel, sr=sr,
        hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap='magma'
    )
    plt.title('Log-Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')

    plt.subplot(1, 2, 2)
    if piano_roll.ndim == 1: # One frame
        plt.bar(np.arange(88), piano_roll, color='purple')
        plt.title('Piano Roll')
        plt.yticks([0, 1])
        plt.xlabel('Piano Key Number')
        plt.ylabel('Activation')
    else: # Multiple frames
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='magma_r')
        plt.title('Piano Roll')
        plt.xlabel('Time Frame')
        plt.ylabel('Piano Key Number')

    plt.show()

def plot_predictions_vs_ground_truth(
    pred_notes: np.ndarray, gt_notes: np.ndarray
) -> None:
    """
    Plot predicted piano roll vs. ground-truth piano roll side by side.

    Parameters
    ----------
    pred_notes : np.ndarray
        A NumPy array of shape (88) or (88, n_frames) representing the
        predicted piano roll.
    gt_notes : np.ndarray
        A NumPy array of shape (88) or (88, n_frames) representing the ground
        truth piano roll.
    """
    if pred_notes.ndim == 1: # One frame
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(88), pred_notes, color='purple')
        plt.title('Predicted Piano Roll')
        plt.yticks([0, 1])
        plt.xlabel('Piano Key Number')
        plt.ylabel('Activation')

        plt.subplot(1, 2, 2)
        plt.bar(np.arange(88), gt_notes, color='purple')
        plt.title('Ground Truth Piano Roll')
        plt.yticks([0, 1])
        plt.xlabel('Piano Key Number')
        plt.ylabel('Activation')
    else: # Multiple frames
        plt.subplot(1, 2, 1)
        plt.imshow(pred_notes, aspect='auto', origin='lower', cmap='magma_r')
        plt.title('Predicted Piano Roll')
        plt.xlabel('Time Frame')
        plt.ylabel('Piano Key Number')

        plt.subplot(1, 2, 2)
        plt.imshow(gt_notes, aspect='auto', origin='lower', cmap='magma_r')
        plt.title('Ground Truth Piano Roll')
        plt.xlabel('Time Frame')
        plt.ylabel('Piano Key Number')

    plt.show()

def plot_training_curves(
    train_loss: np.ndarray, val_loss: np.ndarray,
    train_f1_scores: np.ndarray, val_f1_scores: np.ndarray
) -> None:
    """
    Plot training and validation loss and F1 curves over epochs.

    Parameters
    ----------
    train_loss : np.ndarray
        A 1D NumPy array of training loss values per epoch.
    val_loss : np.ndarray
        A 1D NumPy array of validation loss values per epoch.
    train_f1_scores : np.ndarray
        A 1D NumPy array of training F1 scores per epoch.
    val_f1_scores : np.ndarray
        A 1D NumPy array of validation F1 scores per epoch
    """
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_loss)) + 1, train_loss, label='Train')
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, label='Validation')

    plt.title('Train vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(train_f1_scores)) + 1, train_f1_scores, label='Train')
    plt.plot(np.arange(len(val_f1_scores)) + 1, val_f1_scores, label='Validation')

    plt.title('Train vs. Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')

    plt.show()
