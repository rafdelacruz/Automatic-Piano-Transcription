from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Audio processing parameters
TARGET_SR = 16000 # Hz
HOP_LENGTH = 512
N_MELS = 229