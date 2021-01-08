import pandas as pd
from PCA import PCAAlgorithm


class DataSet:
    def __init__(self, path: str, seperator: str, datasetIndex: int, dimReductionAlgorithm=PCAAlgorithm()):
        self.path = path
        self.seperator = seperator
        self.dataFrame = None
        self.groundTruthColumns = []
        self.groundTruth = None
        self.datasetIndex = datasetIndex
        self.dimReductionAlgorithm = dimReductionAlgorithm

    def _loadCSV(self): # protected
        self.dataFrame = pd.read_csv(self.path, sep=self.seperator)

    def prepareDataset(self):
        pass

    def _reduceDimensions(self):
        self.dimReductionAlgorithm.reduceDimensions(self.dataFrame)
        self.dataFrame = self.dimReductionAlgorithm.getDataFrame()

    def getDataFrame(self) -> pd.DataFrame:
        if self.dataFrame is None:
            self.prepareDataset()
        return self.dataFrame

    def getGroundTruth(self) -> pd.DataFrame: 
        return self.groundTruth

    def getDatasetIndex(self) -> int:
        return self.datasetIndex
