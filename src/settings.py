from enum import Enum


class FeatureExtractorModel(Enum):
    FCN = 0


SEL_FEATURE_EXTRACTOR = FeatureExtractorModel.FCN


# TRAIN_DATASET_PATH_EXTRACTOR_MODEL = 'c:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset/sapimouse_v4_VX_VY_NO_SCALER_3min.csv'
TRAIN_DATASET_PATH_EXTRACTOR_MODEL = 'c:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset/DFL_VX_VY_STANDARD_SCALER_2000.csv'
TRAIN_DATASET_PATH_OCC_MODEL = 'c:/Anaconda projects/afeCAPTCHA/datasets/extracted_features/extracted_features_f128_vx_vy_std_scaler_3min.csv'


TEST_DATASET_PATH_OCC_MODEL = 'c:/Anaconda projects/afeCAPTCHA/datasets/extracted_features/extracted_features_f128_vx_vy_std_scaler_1min.csv'


OUTPUT_MODEL_DIRECTORY = 'C:/Anaconda projects/afeCAPTCHA/trained_models'


OUTPUT_RESULTS_DIRECTORY = 'C:/Anaconda projects/afeCAPTCHA/results'


OUTPUT_EXTRACTED_FEATURES_DIRECTORY = 'C:/Anaconda projects/afeCAPTCHA/datasets/extracted_features'


NB_CLASSES = 120


AGG_BLOCK_NUM = 1


VERBOSE = True

PLOT_TRAIN = True