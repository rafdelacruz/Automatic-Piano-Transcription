from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_SAMPLE_DIR = DATA_DIR / 'sample'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'

# Audio processing parameters
TARGET_SR = 16000 # Hz
HOP_LENGTH = 512
N_MELS = 229