from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from DataSet1 import DataSet1

ds = DataSet1()
ag = AgglomerativeClusteringAlgorithm(nClusters=3, dataFrame=ds.getDataFrame())

print(ag.getSilhouetteScoreList([1,2,3]))