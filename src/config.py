from pathlib import Path

DATASET = 'maestro-v3.0.0'
SPLIT_CONFIGURATION_FILE = 'maestro-v3.0.0.csv'

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_SAMPLE_DIR = DATA_DIR / 'sample'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'

PRIMARY_TRAIN_DIR = DATA_PROCESSED_DIR / 'primary' / 'train'
PRIMARY_VAL_DIR = DATA_PROCESSED_DIR / 'primary' / 'val'
PRIMARY_TEST_DIR = DATA_PROCESSED_DIR / 'primary' / 'test'

BASELINE_TRAIN_DIR = DATA_PROCESSED_DIR / 'baseline' / 'train'
BASELINE_VAL_DIR = DATA_PROCESSED_DIR / 'baseline' / 'val'
BASELINE_TEST_DIR = DATA_PROCESSED_DIR / 'baseline' / 'test'

EXPERIMENTS_DIR = BASE_DIR / 'experiments'
RESULTS_DIR = BASE_DIR / 'result'

# Audio processing parameters
TARGET_SR = 16000 # Hz
HOP_LENGTH = 512
N_MELS = 229
SEGMENT_DURATION = 5.0