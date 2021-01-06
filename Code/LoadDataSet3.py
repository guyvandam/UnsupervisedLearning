from LoadData import LoadData
import pandas as pd
import GlobalParameters
set3Path = GlobalParameters.set3Path


class LoadDataSet3(LoadData):
    def __init__(self, nrows=None):
        super().__init__(set3Path, ";", datasetIndex=3, nrows=nrows)
        self.groundTruthColumns = ['country'] #47 unique values
    
    def prepareDataset(self):
        super().prepareDataset()
        del self.dataFrame['page 2 (clothing model)']
        
        # sub-sampling to 14000 points.
        self.dataFrame = self.dataFrame.sample(n=14000, random_state=GlobalParameters.random_state)
        self.groundTruth = self.dataFrame[self.groundTruthColumns]
        super().reduceDimensions()

       
    
# ld = LoadDataSet3()
# ld.prepareDataset()