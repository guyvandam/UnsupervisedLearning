from sklearn.cluster import AgglomerativeClustering
from ClusterAlgorithm import ClusterAlgorithm
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import GlobalParameters
import numpy as np

class AgglomerativeClusteringAlgorithm(ClusterAlgorithm):

    def __init__(self, nComponents: int, randomState=None, dataFrame=None):
        super().__init__(nComponents, dataFrame=dataFrame)
        self.algorithmObject = AgglomerativeClustering(n_clusters=self.nClusters) # linkage is the distance between clusters.
        self.name = "Agglomerative"

    def getBestNumClusters(self, randomState, numClustersRange, datasetIndex):
        # silhouetteList = []
        # numClustersRange = range(2, numClustersRange+1)
        # for nClusters in numClustersRange:
        #     print(f"Agglomerative clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
        #     self.algorithmObject = AgglomerativeClustering(n_clusters=nClusters)
        #     self.createLabels()
        #     silhouetteList.append(silhouette_score(self.dataFrame, self.labels))

        # plt.figure() # start a new figure.
        # plt.xlabel(GlobalParameters.xlabel)
        # plt.ylabel("Silhouette Score")
        # plt.title("Silhouette Scores for dataset " + str(datasetIndex) + " With Agglomerative Clustering")
        # plt.plot(numClustersRange, silhouetteList, 'bx-')

        # path = GlobalParameters.plotsLocation + str(datasetIndex) + f"\\{self.name}" + str(nClusters) + f"ClustersRandomState{randomState}.png"
        # plt.savefig(path)
        # return numClustersRange[np.argmax(silhouetteList)]
        return super().getBestNumClustersSilhouette(randomState, numClustersRange, datasetIndex)
