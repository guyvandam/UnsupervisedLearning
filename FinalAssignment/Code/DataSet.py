import pandas as pd
from PCA import PCAAlgorithm

class DataSet:
    def __init__(self, path: str, csv_seperator: str, index: int, n_classes: int, dimReductionAlgorithm=PCAAlgorithm()):
        """
        init method.

        Args:
            path (str): path to the CSV file containing the data.
            seperator (str): CSV seperator.
            datasetIndex (int): the data-set index.
            dimReductionAlgorithm (DimRecutionAlgorithm object, optional): dimension redcution algorithm. Defaults to PCAAlgorithm().
        """
        self.csv_file_path = path
        self.csv_seperator = csv_seperator
        self.df = pd.DataFrame()
        # self.ground_truth_columns_list = [] we can drop it as we only have 1 column for ground truth.
        self.ground_truth = None
        self.index = index
        self.dimReductionAlgorithm = dimReductionAlgorithm
        self.n_classes = n_classes # number of classes.

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

    def _reduceDimensions(self):
        """
        reduce the data demension with the dimension reduction algorithm.
        """
        self.dimReductionAlgorithm.reduceDimensions(self.df)
        self.df = self.dimReductionAlgorithm.getDataFrame()

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
