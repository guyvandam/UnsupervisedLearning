import GlobalParameters
import os

def get_file_path(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    return file_path
    
def get_folder_path(folder_name, enclosing_path = None):
    if enclosing_path == None: enclosing_path = os.getcwd()

    folder_path = os.path.join(enclosing_path, folder_name)
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    return folder_path

def get_results_folder_path():
    return get_folder_path(GlobalParameters.RESULTS_FOLDER_NAME)

def get_dataset_folder_name(dataset_index):
    return f"{GlobalParameters.RESULTS_DATASET_FOLDER_NAME}{dataset_index}"

def get_datasets_CSV_files_folder_path():
    return get_folder_path(GlobalParameters.DATASETS_CSV_FILES_FOLDER_NAME)

def get_dataset_CSV_file_path(index):
    files_folder_path = get_datasets_CSV_files_folder_path()
    file_name = GlobalParameters.DATASET_CSV_FILE_NAME_DICT[index]
    path = os.path.join(files_folder_path, file_name)
    
    if os.path.isfile(path):
        return path
    else:
        print("no file")

def get_dataset_n_classes(index):
    return GlobalParameters.DATASETS_NUMBER_OF_CLASSES_DICT[index]

def get_plot_file_path(file_name, dataset_index, parent_folder_name):
    results_folder_path = get_results_folder_path()
    parent_folder_path = get_folder_path(folder_name = parent_folder_name, enclosing_path = results_folder_path)
    dataset_folder_name = get_dataset_folder_name(dataset_index)
    dataset_folder_path = get_folder_path(dataset_folder_name, enclosing_path = parent_folder_path)

    plot_file_path = get_file_path(file_name, dataset_folder_path)
    return plot_file_path