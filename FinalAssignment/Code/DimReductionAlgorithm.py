class DimReductionAlgorithm:

    def __init__(self, nComponents: int):
        """
        init method.

        Args:
            nComponents (int): number of components/dimensions we want the data to be.
        """
        self.nComponents = nComponents
        self.dataFrame = None
        self.algorithmObject = None
        self.dims = ['dim1', 'dim2']

    def reduceDimensions(self, dataFrame):
        """
        Interface method.
        
        Args:
            dataFrame (pandas.DataFrame): the dataFrame containing the data we want to reduce.
        """
        pass

    def getDataFrame(self):
        """
        Retruns the dataFrame containing the data we want to reduce.

        Returns:
            pandas.DataFrame: the dataFrame containing the data we want to reduce.
        """
        return self.dataFrame
