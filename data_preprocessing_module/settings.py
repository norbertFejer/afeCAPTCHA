from enum import Enum


class RawDataType(Enum):
    DX_DY = 0
    ABS_DX_DY = 1
    VX_VY = 2
    ABS_VX_VY = 3


class ScalerType(Enum):
    NO_SCALER = 0
    STANDARD_SCALER = 1


# DATASET_PATH = 'c:/Anaconda projects/afeCAPTCHA/datasets/sapimouse_v4'
DATASET_PATH = 'c:/Anaconda projects/Software_mod/DFL'

PREPROCESSED_DATASET_PATH = 'C:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset'

INPUT_EXTRACT_FEATURES_PATH = 'c:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset/sapimouse_v4_VX_VY_STANDARD_SCALER_1min.csv'

OUTPUT_EXTRACTED_FEATURES_PATH  = 'C:/Anaconda projects/afeCAPTCHA/datasets/extracted_features'

STATELESS_TIME = 1000

MIN_MOUSE_EVENT_COUNT = 6

MAX_BLOCK_NUM = 2000

BLOCK_SIZE = 128

SEL_RAW_DATA_TYPE = RawDataType.VX_VY

SEL_SCALER = ScalerType.STANDARD_SCALER

NB_CLASSES = 120


# it is important for reading more rows than required
# because of the boundaries of movements which will be dropped out
ROW_NUM_TOL = 5000