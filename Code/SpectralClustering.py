from sklearn.cluster import SpectralClustering
from ClusterAlgorithm import ClusterAlgorithm
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import GlobalParameters
import numpy as np

class SpectralClusteringAlgorithm(ClusterAlgorithm):

    def __init__(self, nComponents: int, randomState=None, dataFrame=None):
        super().__init__(nComponents, randomState, dataFrame=dataFrame)
        self.algorithmObject = SpectralClustering(n_clusters=self.nClusters, random_state=self.randomState)
        self.name = "Spectral"
    
    def getBestNumClusters(self, randomState, numClustersRange, datasetIndex):
        # silhouetteList = []
        # numClustersRange = range(2, numClustersRange+1)
        # for nClusters in numClustersRange:
        #     print(f"Spectral clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
        #     self.algorithmObject = SpectralClustering(n_clusters=nClusters, random_state=randomState)
        #     self.createLabels()
        #     silhouetteList.append(silhouette_score(self.dataFrame, self.labels))

        # plt.figure() # start a new figure.
        # plt.xlabel(GlobalParameters.xlabel)
        # plt.ylabel("Silhouette Score")
        # plt.title("Silhouette Scores for dataset " + str(datasetIndex) + " With Spectral Clustering")
        # plt.plot(numClustersRange, silhouetteList, 'bx-')

        # path = GlobalParameters.plotsLocation + str(datasetIndex) + "\\SilhouetteScoreSpectralFor" + str(nClusters) + "Clusters.png"
        # plt.savefig(path)
        # return numClustersRange[np.argmax(silhouetteList)]
        return super().getBestNumClustersSilhouette(randomState, numClustersRange, datasetIndex)
    