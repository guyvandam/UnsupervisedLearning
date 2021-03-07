import GlobalParameters
import os

    
def get_folder_path(folder_name, enclosing_path = None):
    if enclosing_path == None: enclosing_path = os.getcwd()

    folder_path = os.path.join(enclosing_path, folder_name)
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    return folder_path

def get_results_folder_path():
    get_folder_path(GlobalParameters.RESULTS_FOLDER_NAME)

def get_datasets_CSV_files_folder_path():
    get_folder_path(GlobalParameters.DATASETS_CSV_FILES_FOLDER_NAME)

def get_dataset_CSV_file_path(index):
    files_folder_path = get_datasets_CSV_files_folder_path()
    path = os.path.join(files_folder_path, GlobalParameters.DATASET_CSV_FILE_NAME_DICT[index])
    
    if os.path.isfile(path):
        return path
    else:
        print("no file")

def get_dataset_n_classes(index):
    return GlobalParameters.DATASETS_NUMBER_OF_CLASSES_DICT[index]
