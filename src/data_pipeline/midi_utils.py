import pathlib

import numpy as np
import pretty_midi

def load_midi(filepath: pathlib.Path) -> pretty_midi.PrettyMIDI:
    """
    Load a MIDI file and parse it into a PrettyMIDI object.

    Parameters
    ----------
    filepath : pathlib.Path
        A Path object pointing to the MIDI file.

    Returns
    -------
    pretty_midi.PrettyMIDI
        A PrettyMIDI object representing the MIDI file (contains all MIDI data).
    """
    midi_data = pretty_midi.PrettyMIDI(filepath)
    return midi_data

def convert_midi_to_piano_roll(
    midi: pretty_midi.PrettyMIDI, n_frames: int, frame_duration: float
) -> np.ndarray:
    """
    Convert a PrettyMIDI object to a piano-roll representation.

    Parameters
    ----------
    midi : pretty_midi.PrettyMIDI
        A PrettyMIDI object representing the MIDI file to convert.
    n_frames : int
        The number of time frames in the resulting piano roll.
    frame_duration : float
        The duration of each frame (in seconds).

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (128, n_frames) representing the piano roll,
        where rows correspond to MIDI pitches (0-127) and columns correspond
        to time frames.
    """
    piano_roll = np.zeros((128, n_frames), dtype=np.float32)

    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                start_frame = int(np.round(note.start / frame_duration))
                end_frame = int(np.round(note.end / frame_duration))

                if end_frame > n_frames:
                    end_frame = n_frames

                piano_roll[note.pitch, start_frame:end_frame] = 1.0

    return piano_roll

def convert_piano_roll_to_midi(
    piano_roll: np.ndarray,
    sr: float,
    hop_length: int=512,
    default_velocity: int=100,
    program: int=0
) -> pretty_midi.PrettyMIDI:
    """
    Convert a piano-roll array to a PrettyMIDI object.

    Parameters
    ----------
    piano_roll : np.ndarray
        A 2D NumPy array of shape (128, n_frames) representing the piano roll.
    sr : float
        The sampling rate of the original audio (in Hz).
    hop_length : int, default=512
        The number of audio samples per piano-roll frame.
    default_velocity : int, default=100
        The default velocity (MIDI intensity, 1-127) when a note is active.
    program: int, default=0
        The MIDI program number (corresponds to an instrument, 0-127).
        0 = Acoustic Grand Piano.
    
    Returns
    -------
    pretty_midi.PrettyMIDI
        A PrettyMIDI object that represents the piano roll array.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program)

    frame_duration = hop_length / sr
    num_frames = piano_roll.shape[1]

    for pitch in range(128):
        roll = piano_roll[pitch, :]
        roll = (roll > 0.5).astype(np.int8)

        padded = np.pad(roll, (1, 1))
        changes = np.diff(padded)
        note_on_frames = np.where(changes == 1)[0]
        note_off_frames = np.where(changes == -1)[0]

        for on, off in zip(note_on_frames, note_off_frames):
            start = on * frame_duration
            end = off * frame_duration

            note = pretty_midi.Note(
                default_velocity,
                pitch,
                start,
                end
            )

            instrument.notes.append(note)

    pm.instruments.append(instrument)
    
    return pm
