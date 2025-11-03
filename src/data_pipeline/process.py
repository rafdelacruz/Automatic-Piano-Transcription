import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi

# Parameters
wav_file = './data/sample/wav_test.wav'
midi_file = './data/sample/midi_test.midi'
target_sr = 16000
hop_length = 512
n_mels = 229

# Load .wav and .midi files
audio, sr = librosa.load(wav_file, sr=target_sr)
midi_data = pretty_midi.PrettyMIDI(midi_file)

# Calculate other parameters
audio_duration = librosa.get_duration(y=audio, sr=sr)
frame_duration = hop_length / sr
n_frames = int(np.ceil(audio_duration / frame_duration))

# Create piano roll
piano_roll = np.zeros((128, n_frames), dtype=np.float32)

for instrument in midi_data.instruments:
    if not instrument.is_drum:
        for note in instrument.notes:
            start_frame = int(np.round(note.start / frame_duration))
            end_frame = int(np.round(note.end / frame_duration))

            if end_frame > n_frames:
                end_frame = n_frames

            piano_roll[note.pitch, start_frame:end_frame] = 1.0

# Create log-mel spectrogram
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
log_mel = librosa.power_to_db(mel, ref=np.max)

print("Audio spectrogram shape:", log_mel.shape)
print("MIDI piano roll shape:", piano_roll.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    log_mel,
    sr=sr,
    hop_length=hop_length,
    x_axis='time',
    y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Log-Mel Spectrogram")
plt.tight_layout()
plt.show()
