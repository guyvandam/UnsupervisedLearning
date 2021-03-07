import pandas as pd
from PCA import PCAAlgorithm
import GlobalFunctions

class DataSet:
    def __init__(self, index: int, csv_seperator: str = ','):
        """
        init method.

        Args:
            path (str): path to the CSV file containing the data.
            seperator (str): CSV seperator.
            datasetIndex (int): the data-set index.
            dimReductionAlgorithm (DimRecutionAlgorithm object, optional): dimension redcution algorithm. Defaults to PCAAlgorithm().
        """
        self.index = index
        self.csv_seperator = csv_seperator
        self.csv_file_path = GlobalFunctions.get_dataset_CSV_file_path(self.index)
        self.n_classes = GlobalFunctions.get_dataset_n_classes(self.index) # number of classes.
        
        self.df = pd.DataFrame()
        self.ground_truth = None

    def _loadCSV(self, na_values=None):
        """
        Protected
        Load the CSV file at the path location into a pandas DataFrame
        """
        self.df = pd.read_csv(self.csv_file_path, sep=self.csv_seperator, na_values=na_values)

    def prepareDataset(self):
        """
        Interface method
        prepare the dataset for clustering.
        """
        pass

    def _reduceDimensions(self, dim_reduction_algorithm_obj = PCAAlgorithm()):
        """
        reduce the data demension with the dimension reduction algorithm.
        """
        dim_reduction_algorithm_obj.reduceDimensions(self.df)
        self.df = dim_reduction_algorithm_obj.getDataFrame()

    def get_data_frame(self) -> pd.DataFrame:
        """
        returns the DataFrame represention of the data-set.

        Returns:
            pd.DataFrame: DataFrame represention of the data-set.
        """
        if self.df.empty:
            self.prepareDataset()
        return self.df

    def get_ground_truth(self) -> pd.DataFrame:
        """
        Returns the ground truth labels for the datasets.

        Returns:
            pd.DataFrame: ground truth DataFrame
        """
        return self.ground_truth

    def get_index(self) -> int:
        """
        Returns the data-set index

        Returns:
            int: the data-set index
        """
        return self.index

    def get_n_classes(self) -> int:
        """
        Returns:
            int: n_class - number of classes for the dataset.
        """
        return self.n_classes

    def drop_rows_by_non_na_precent(self, non_na_precent):
        ################################ thresh -  Require that many non-NA values.
        num_columns = len(self.df.columns)
        precent_fraction = float(non_na_precent / 100)
        non_na_number_rows = int(num_columns * precent_fraction)
        self.df.dropna(axis = 0, thresh = int(non_na_number_rows), inplace = True)

