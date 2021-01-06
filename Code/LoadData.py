import pandas as pd
from PCA import PCAAlgorithm

class LoadData(object):
    def __init__(self, path: str, seperator: str, datasetIndex, nrows=None, dimReductionAlgorithm=None):
        self.path = path
        self.seperator = seperator
        self.dataFrame = None
        self.nrows = nrows
        # self.groundTruthColumns = ['VisitorType', 'Weekend', 'Revenue']
        self.groundTruthColumns = []
        self.groundTruth = None
        self.datasetIndex = datasetIndex
        self.dimReductionAlgorithm = dimReductionAlgorithm

    def __loadCSV(self):
        self.dataFrame = pd.read_csv(
            self.path, sep=self.seperator, nrows=self.nrows)

    def printHead(self):
        print(self.dataFrame.head())

    def prepareDataset(self):
        self.__loadCSV()

    def reduceDimensions(self):
        if self.dimReductionAlgorithm is None:
            self.dimReductionAlgorithm = PCAAlgorithm()
        self.dimReductionAlgorithm.reduceDimensions(self.dataFrame)
        self.dataFrame = self.dimReductionAlgorithm.getDataFrame()

    def getDataFrame(self) -> pd.DataFrame:
        if self.dataFrame is None:
            self.prepareDataset()
        return self.dataFrame

    def getGroundTruth(self): # list of lists
        # if self.dataFrame is None:
        #     self.prepareDataset()
        # result = self.dataFrame[self.groundTruthColumns]
        # return result
        return self.groundTruth

    def getDatasetIndex(self):
        return self.datasetIndex
