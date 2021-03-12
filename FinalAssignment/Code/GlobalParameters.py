import os
global DATASET1_NUMBER_OF_CLASSES, DATASET2_NUMBER_OF_CLASSES, randomStateList, randomState


RESULTS_FOLDER_NAME = "Results"

EXTERNAL_LABELS_FITTMENT_RESULTS_FOLDER = "ExternalLabelsFittment"

OPTIMAL_N_CLUSTERS_FOLDER_NAME = "OptimalNClusters"

RESULTS_DATASET_FOLDER_NAME = "Dataset"

CLUSTERING_PLOT_FOLDER_NAME = "ClusteringPlot"

STATISTICAL_TEST_FOLDER_NAME = "StatisticalTest"


############################## datasets csv files folder
DATASETS_CSV_FILES_FOLDER_NAME = "Datasets"

############################## datasets csv file names
DATASET1_CSV_FILE_NAME = "allUsers.lcl.csv"
DATASET2_CSV_FILE_NAME = "HTRU_2.csv"

DATASET_CSV_FILE_NAME_DICT = {1: DATASET1_CSV_FILE_NAME, 2: DATASET2_CSV_FILE_NAME}

# ############################## dataset number of classes
# DATASET1_NUMBER_OF_CLASSES = 5
# DATASET2_NUMBER_OF_CLASSES = 2

# DATASETS_NUMBER_OF_CLASSES_DICT = {1:DATASET1_NUMBER_OF_CLASSES, 2:DATASET2_NUMBER_OF_CLASSES}

############################## random state
random_state = 0
randomStateList = [68, 123, 4255, 17, 1, 63, 11, 77, 4, 453, 537, 5323, 3, 42, 323, 21, 617, 7, 0, 1234, 9, 61, 57, 583, 19, 12345, 3,
                   14, 4257, 5]
