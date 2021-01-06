import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize # ----------------- need to add normalization
from DimReductionAlgorithm import DimReductionAlgorithm
import GlobalParameters
class PCAAlgorithm(DimReductionAlgorithm):

    def __init__(self,nComponents:int=2):
        super().__init__(nComponents)
        self.algorithmObject = PCA(n_components=self.nComponents, random_state=GlobalParameters.random_state)

    def normalize(self):
        self.dataFrame = StandardScaler().fit_transform(self.dataFrame)
        self.dataFrame = normalize(self.dataFrame)

    def reduceDimensions(self,dataFrame):
        super().reduceDim(dataFrame)
        self.normalize()
        returnMatrix = self.algorithmObject.fit_transform(self.dataFrame) # (n_samples, n_components) changes in dim1 are more important than in dim2
        self.dataFrame = pd.DataFrame(data=returnMatrix,columns=self.dims)

    def getDataFrame(self):
        return self.dataFrame
     
