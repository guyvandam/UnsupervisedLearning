class DimReductionAlgorithm():

    def __init__(self, nComponents:int):
        self.nComponents = nComponents
        self.dataFrame = None
        self.algorithmObject = None
        self.dims = ['dim1','dim2']
    
    def reduceDim(self,dataFrame):
        self.dataFrame = dataFrame
        """[
            manipulates the dataFrame
            changes it into a 2 dim dataFrame
        ]
        """
        print("Reducing Dimensions")

    def getDataFrame(self):
        return self.dataFrame