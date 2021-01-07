# from KMeans import KMeansAlgorithm
from LoadDataSet1 import LoadDataSet1
from LoadDataSet3 import LoadDataSet3
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from KMeans import KMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm

ld = LoadDataSet3()
ld.prepareDataset()

calgo = KMeansAlgorithm()

calgo = GMMAlgorithm()
# calgo = FuzzyCMeansAlgorithm(nComponents=1)
# calgo = AgglomerativeClusteringAlgorithm(nComponents=1)
# calgo = SpectralClusteringAlgorithm(nComponents=1,dataFrame=ld.getDataFrame())
calgo.setDataFrame(ld.getDataFrame())
print(calgo.getBestNumClusters(42,8,ld.datasetIndex))
