from sklearn.metrics import silhouette_samples, silhouette_score
from LoadAndPlot import DoAll
from GMM import GMMAlgorithm
from PCA import PCAAlgorithm
from LoadDataSet3 import LoadDataSet3
from KMeans import KMeansAlgorithm
from FuzzyCMeans import FuzzyCMeans
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm 
from SpectralClustering import SpectralClusteringAlgorithm
# from SilhouetteScore import SilhouetteScore
import numpy as np

class SilhouetteScore():

    def __init__(self,dataFrame,labels):

        self.dataFrame = dataFrame
        self.labels = labels
        self.silhouetteScore = None
    
    def createSilhouetteScore(self):
        self.silhouetteScore = silhouette_score(self.dataFrame, self.labels)


    # def plotSilhouetteScore(self, maxClusters:int):
    #     kmeans = 
    #     da = DoAll()
    #     for nClusters in range(1,maxClusters):
    #         silhouette_avg = silhouette_score(X, cluster_labels)
    #         print("For n_clusters =", nClusters, "The average silhouette_score is :", silhouette_avg)


clusterComponents = 3
pcaComp = 2

gmm = GMMAlgorithm(clusterComponents)
pca = PCAAlgorithm()

loadData = LoadDataSet3(nrows=100)


nr = range(1,4)
nr = [2,3,4]
for n in nr:
    kmeans = KMeansAlgorithm(n)
    da = DoAll(loadData,kmeans,pca)
    

    da.initialize()
    da.createLabels()
    labels = da.getLabels()

    silhouette_avg = silhouette_score(da.loadData.getDataFrame(), labels)
    print("For n_clusters =", n,
          "The average silhouette_score is :", silhouette_avg)

 
