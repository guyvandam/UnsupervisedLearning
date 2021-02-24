from DataSet import DataSet
import GlobalParameters

set3Path = GlobalParameters.set3Path


class DataSet3(DataSet):
    def __init__(self):
        """
        init method.
        ground truth labels were given to us in the assignment page.
        
        """
        super().__init__(set3Path, ";", datasetIndex=3)
        self.groundTruthColumns = ['country']  # 47 unique values, about 36 after sampling.

    def prepareDataset(self):
        """
        Inherited Function
        delete irrelevant column, sample 14000 points using pandas' sample() method with a fixed random state.
        
        """
        super()._loadCSV()
        del self.dataFrame['page 2 (clothing model)']

        # sub-sampling to 14000 points.
        self.dataFrame = self.dataFrame.sample(n=14000, random_state=GlobalParameters.randomState)
        self.groundTruth = self.dataFrame[self.groundTruthColumns]
        super()._reduceDimensions()
