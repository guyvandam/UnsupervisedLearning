import pandas as pd
from PCA import PCAAlgorithm


class DataSet:
    def __init__(self, path: str, seperator: str, datasetIndex: int, dimReductionAlgorithm=PCAAlgorithm()):
        """
        init method.

        Args:
            path (str): path to the CSV file containing the data.
            seperator (str): CSV seperator.
            datasetIndex (int): the data-set index.
            dimReductionAlgorithm (DimRecutionAlgorithm object, optional): dimension redcution algorithm. Defaults to PCAAlgorithm().
        """
        self.path = path
        self.seperator = seperator
        self.dataFrame = None
        self.groundTruthColumns = []
        self.groundTruth = None
        self.datasetIndex = datasetIndex
        self.dimReductionAlgorithm = dimReductionAlgorithm

    def _loadCSV(self):
        """
        Protected
        Load the CSV file at the path location into a pandas DataFrame
        """
        self.dataFrame = pd.read_csv(self.path, sep=self.seperator)

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
        self.dimReductionAlgorithm.reduceDimensions(self.dataFrame)
        self.dataFrame = self.dimReductionAlgorithm.getDataFrame()

    def getDataFrame(self) -> pd.DataFrame:
        """
        returns the DataFrame represention of the data-set.

        Returns:
            pd.DataFrame: DataFrame represention of the data-set.
        """
        if self.dataFrame is None:
            self.prepareDataset()
        return self.dataFrame

    def getGroundTruth(self) -> pd.DataFrame:
        """
        Returns the ground truth labels for the datasets.

        Returns:
            pd.DataFrame: ground truth DataFrame
        """
        return self.groundTruth

    def getDatasetIndex(self) -> int:
        """
        Returns the data-set index

        Returns:
            int: the data-set index
        """
        return self.datasetIndex
