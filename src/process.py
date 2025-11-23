import pathlib

import librosa
import numpy as np
import pandas as pd

import config
from data_pipeline.audio_utils import load_audio, compute_log_mel_spectrogram
from data_pipeline.midi_utils import load_midi, convert_midi_to_piano_roll
from data_pipeline.segmentation import segment_pair_primary, segment_pair_baseline

def process_pair(
    wav_path: pathlib.Path, midi_path: pathlib.Path,
    target_sr: float, n_mels: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a pair of audio and MIDI files into model-ready features.

    This function performs the preprocessing steps on a pair of audio and MIDI
    files. The log-Mel spectrogram is computed for the audio and a piano-roll
    is generated for the MIDI. The resultng arrays are aligned in time based
    on the audio's duration and the hop length.

    Parameters
    ----------
    wav_path : pathlib.Path
        A Path object pointing to the audio file.
    midi_path : pathlib.Path
        A Path object pointing to the MIDI file.
    target_sr : float
        The target sampling rate to resample the audio to (in Hz).
    n_mels : int
        The number of Mel bands to generate.
    hop_length : int
        The number of samples between successive frames.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - np.ndarray
            The log-Mel spectrogram (in decibels) of shape (n_mels, n_frames).
        - np.ndarray
            The piano-roll representation of shape (128, n_frames).
    """
    # Load audio and midi files
    audio_data = load_audio(wav_path, target_sr=target_sr)
    midi_data = load_midi(midi_path)

    # Compute sub-parameters
    audio_duration = librosa.get_duration(y=audio_data, sr=target_sr)
    frame_duration = hop_length / target_sr
    n_frames = int(np.ceil(audio_duration / frame_duration))

    # Compute log-Mel spectrogram + generate piano roll
    log_mel = compute_log_mel_spectrogram(
                audio_data, target_sr,
                n_mels=n_mels, hop_length=hop_length,
    )
    piano_roll = convert_midi_to_piano_roll(midi_data, n_frames, frame_duration)

    return log_mel, piano_roll

def process_dataset(dataset: str, split_file: str, primary: bool = True) -> None:
    """
    Process the specified dataset and save the processed data to appropriate
    directories for training, validation, or testing.

    This function expects a metadata CSV file containing the following columns:
        - 'audio_filename': Name of the audio file (usually .wav file).
        - 'midi_filename': Name of the corresponding MIDI file (usually .midi file).
        - 'split': A string indicating whether the pair belongs to the train,
            validation, or test split.

    Each pair of audio and MIDI files is sent through the data processing
    pipeline to generate spectrogram and piano roll features and is then
    segmented into smaller chunks. The processed data is saved to the
    corresponding split.
    
    Parameters
    ----------
    dataset : str
        A string holding the name of the dataset.
    split_file : str
        A string holding the .csv file containing the dataset split.
    primary : bool, default=True
        Indicates whether data is being processed for the primary model.
    """
    dataset_dir = config.DATA_RAW_DIR / dataset
    metadata_df = pd.read_csv(dataset_dir / split_file)

    # Each array stores [[log-mel segments], [piano_roll segments]]
    train_segments = [[], []]
    val_segments = [[], []]
    test_segments = [[], []]

    train_dir = config.PRIMARY_TRAIN_DIR if primary else config.BASELINE_TRAIN_DIR
    val_dir = config.PRIMARY_VAL_DIR if primary else config.BASELINE_VAL_DIR
    test_dir = config.PRIMARY_TEST_DIR if primary else config.BASELINE_TEST_DIR

    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(metadata_df)):
        metadata = metadata_df.iloc[i]
        wav_path = dataset_dir / metadata['audio_filename']
        midi_path = dataset_dir / metadata['midi_filename']
        split = metadata['split']

        # Generate log_mel spectrogram and piano roll
        log_mel, piano_roll = process_pair(
            wav_path, midi_path,
            config.TARGET_SR,   
            config.N_MELS,
            config.HOP_LENGTH
        )

        # Segment spectrogram and piano roll
        if primary:
            log_mel_segments, piano_roll_segments = segment_pair_primary(
                log_mel, piano_roll,
                config.TARGET_SR,
                hop_length=config.HOP_LENGTH,
                segment_duration=config.SEGMENT_DURATION
            )
        else:
            log_mel_segments, piano_roll_segments = segment_pair_baseline(
                log_mel, piano_roll
            )

        if split == 'train':
            train_segments[0].extend(log_mel_segments)
            train_segments[1].extend(piano_roll_segments)
        
        elif split == 'validation':
            val_segments[0].extend(log_mel_segments)
            val_segments[1].extend(piano_roll_segments)
        
        else:
            test_segments[0].extend(log_mel_segments)
            test_segments[1].extend(piano_roll_segments)

        if i % 10 == 0:
            print(f'{i + 1}/{len(metadata_df)} samples processed!')

    # Save the data pairs to their respective folder
    np.save(train_dir / 'train_log_mel_segments.npy', np.stack(train_segments[0]))
    np.save(train_dir / 'train_piano_roll_segments.npy', np.stack(train_segments[1]))
    np.save(val_dir / 'val_log_mel_segments.npy', np.stack(val_segments[0]))
    np.save(val_dir / 'val_piano_roll_segments.npy', np.stack(val_segments[1]))
    np.save(test_dir / 'test_log_mel_segments.npy', np.stack(test_segments[0]))
    np.save(test_dir / 'test_piano_roll_segments.npy', np.stack(test_segments[1]))

    print('Segments saved!')

def process_and_save_sample(
    wav_path: pathlib.Path, midi_path: pathlib.Path, primary: bool = False
) -> None:
    """
    Process an audio sample and its corresponding MIDI data, and save it to the
    sample directory.

    This function sends the audio and MIDI files through the data processing
    pipeline to generate spectrogram and piano roll features and is then
    segmented into smaller chunks.

    Parameters
    ----------
    wav_sample_path : pathlib.Path
        A Path object pointing to the audio file to process.
    midi_sample_path : pathlib.Path
        A Path object pointing to the MIDI file to process.
    
    """
    # Generate log_mel spectrogram and piano roll
    log_mel, piano_roll = process_pair(
        wav_path, midi_path,
        config.TARGET_SR,   
        config.N_MELS,
        config.HOP_LENGTH
    )

    # Segment spectrogram and piano roll
    if primary:
        log_mel_segments, piano_roll_segments = segment_pair_primary(
            log_mel, piano_roll,
            config.TARGET_SR,
            hop_length=config.HOP_LENGTH,
            segment_duration=config.SEGMENT_DURATION
        )
    else:
        log_mel_segments, piano_roll_segments = segment_pair_baseline(
            log_mel, piano_roll
        )

    # Save the data pairs to their respective folder
    np.save(config. DATA_SAMPLE_DIR / 'log_mel_segments.npy', log_mel_segments)
    np.save(config. DATA_SAMPLE_DIR / 'piano_roll_segments.npy', piano_roll_segments)

    print('Segments saved!')

if __name__ == '__main__':
    process_dataset('maestro-v3.0.0', 'maestro-v3.0.0.csv', primary=False)
