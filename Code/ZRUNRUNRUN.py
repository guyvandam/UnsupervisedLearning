from LoadAndPlot import DoAll
from GMM import GMMAlgorithm
from PCA import PCAAlgorithm
from LoadDataSet3 import LoadDataSet3
from KMeans import KMeansAlgorithm
from FuzzyCMeans import FuzzyCMeans
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm 
from SpectralClustering import SpectralClusteringAlgorithm
from SilhouetteScore import SilhouetteScore
import numpy as np

def checkAgainst(l1, l2):
    if len(l1) == len(l2):
        return np.mean(l1==l2)
    print("error")

clusterComponents = 3
pcaComp = 2

gmm = GMMAlgorithm(clusterComponents)
kmeans = KMeansAlgorithm(clusterComponents)
pca = PCAAlgorithm(pcaComp)
agglo = AgglomerativeClusteringAlgorithm(clusterComponents)
fuzzy = FuzzyCMeans(clusterComponents)
spec = SpectralClusteringAlgorithm(clusterComponents)
loadData = LoadDataSet3(nrows=100)

# da = DoAll(loadData,fuzzy,pca)
# da = DoAll(loadData,gmm,pca)
# da = DoAll(loadData,agglo,pca)
da = DoAll(loadData,kmeans,pca)
# da = DoAll(loadData,spec,pca)
da.plotData()
X = loadData.dataFrame
print(X)
# sill = SilhouetteScore(X,da.getLabels())
# sill.createSilhouetteScore()
# print(sill.silhouetteScore)

"""checking againset the classifier"""

classifier3 = 'country'
real = X[classifier3]
print(real)
labels = da.getLabels()
print(labels)
print(checkAgainst(real, labels))
