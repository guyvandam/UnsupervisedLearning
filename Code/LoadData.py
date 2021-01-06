import pandas as pd
from PCA import PCAAlgorithm


class LoadData():
    def __init__(self, path: str, seperator: str, datasetIndex: int, nrows: int = None, dimReductionAlgorithm=None):
        self.path = path
        self.seperator = seperator
        self.dataFrame = None
        self.nrows = nrows
        self.groundTruthColumns = []
        self.groundTruth = None
        self.datasetIndex = datasetIndex
        self.dimReductionAlgorithm = dimReductionAlgorithm

    def _loadCSV(self):
        self.dataFrame = pd.read_csv(
            self.path, sep=self.seperator, nrows=self.nrows)

    def prepareDataset(self):
        pass

    def reduceDimensions(self):
        if self.dimReductionAlgorithm is None:
            self.dimReductionAlgorithm = PCAAlgorithm()
        self.dimReductionAlgorithm.reduceDimensions(self.dataFrame)
        self.dataFrame = self.dimReductionAlgorithm.getDataFrame()

    def getDataFrame(self) -> pd.DataFrame:
        if self.dataFrame is None:
            self.prepareDataset()
        return self.dataFrame

    def getGroundTruth(self) -> pd.DataFrame:
        return self.groundTruth

    def getDatasetIndex(self):
        return self.datasetIndex
