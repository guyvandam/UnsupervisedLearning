from DataSet import DataSet
import pandas as pd
import GlobalParameters
set1Path = GlobalParameters.set1Path


class DataSet1(DataSet):

    def __init__(self, nrows=None):
        super().__init__(path=set1Path, seperator=",", datasetIndex=1, nrows=nrows)
        self.groundTruthColumns = ['VisitorType', 'Weekend', 'Revenue']
    
    def prepareDataset(self):
        """not the best approach to hardcode it in, but after inspecting the dataset, it's not the worst idea"""

        monthNumberDict = {
            'Jan': 1,
            'Feb': 2,
            'Mar': 3,
            'Apr': 4,
            'May': 5,
            'June': 6,
            'Jul': 7,
            'Aug': 8,
            'Sep': 9,
            'Oct': 10,
            'Nov': 11,
            'Dec': 12
        }

        visitorTypeDict = {
            'New_Visitor': 0,
            'Returning_Visitor': 1,
            'Other': 2
        }

        self.dataFrame = pd.read_csv(
            self.path, sep=self.seperator, nrows=self.nrows, converters={'Month': lambda x: monthNumberDict[x], 'VisitorType': lambda x: visitorTypeDict[x]})
        trueFalseColumns = ['Weekend', 'Revenue']
        for c in trueFalseColumns:
            self.dataFrame[c] = self.dataFrame[c].astype(int)
            
        # 12330 points
        self.groundTruth = self.dataFrame[self.groundTruthColumns]
        super()._reduceDimensions()

# ld = LoadDataSet1()
# ld.prepareDataset()
