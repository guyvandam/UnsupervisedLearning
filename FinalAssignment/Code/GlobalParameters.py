import os
global DATASET1_NUMBER_OF_CLASSES, DATASET2_NUMBER_OF_CLASSES, randomStateList, randomState


RESULTS_FOLDER_NAME = "Results"

EXTERNAL_LABELS_FITTMENT_RESULTS_FOLDER = "ExternalLabelsFittment"

RESULTS_DATASET_FOLDER_NAME = "Dataset"

############################## datasets csv files folder
DATASETS_CSV_FILES_FOLDER_NAME = "Datasets"

############################## datasets csv file names
DATASET1_CSV_FILE_NAME = "allUsers.lcl.csv"
DATASET2_CSV_FILE_NAME = "HTRU_2.csv"

DATASET_CSV_FILE_NAME_DICT = {1: DATASET1_CSV_FILE_NAME, 2: DATASET2_CSV_FILE_NAME}

############################## dataset number of classes
DATASET1_NUMBER_OF_CLASSES = 5
DATASET2_NUMBER_OF_CLASSES = 2

DATASETS_NUMBER_OF_CLASSES_DICT = {1:DATASET1_NUMBER_OF_CLASSES, 2:DATASET2_NUMBER_OF_CLASSES}

############################## random state
random_state = 42
randomStateList = [68, 94, 60, 17, 1, 63, 11, 77, 4, 45, 24, 15, 3, 42, 32, 21, 62, 7, 0, 78, 9, 61, 57, 84, 19, 56, 2,
                   14, 87, 46]
