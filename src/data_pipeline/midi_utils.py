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
