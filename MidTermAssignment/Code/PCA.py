import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from DimReductionAlgorithm import DimReductionAlgorithm
import GlobalParameters


class PCAAlgorithm(DimReductionAlgorithm):

    def __init__(self, nComponents: int = 2):
        """
        init method.

        Args:
            nComponents (int, optional): number of components/dimensions we want the data to be. Defaults to 2.
        """
        super().__init__(nComponents)
        self.algorithmObject = PCA(n_components=self.nComponents, random_state=GlobalParameters.randomState)

    def normalize(self):
        """
        Normalize our data with the fit_transform() method from the StandardScaler object and normalize function from sklearn.preprocessing
        
        """
        self.dataFrame = StandardScaler().fit_transform(self.dataFrame)
        self.dataFrame = normalize(self.dataFrame)

    def reduceDimensions(self, dataFrame):
        """
        Reduce the dataFrame dimensions to nComponents using fit_transform method in the PCA object.
        Args:
            dataFrame (pandas.DataFrame): the dataFrame containing the data we want to reduce.
        """
        self.dataFrame = dataFrame
        self.normalize()
        returnMatrix = self.algorithmObject.fit_transform(
            self.dataFrame)  # (n_samples, n_components) changes in dim1 are more important than in dim2
        self.dataFrame = pd.DataFrame(data=returnMatrix, columns=self.dims)
