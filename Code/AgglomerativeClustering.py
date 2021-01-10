from sklearn.cluster import AgglomerativeClustering

from ClusterAlgorithm import ClusterAlgorithm


class AgglomerativeClusteringAlgorithm(ClusterAlgorithm):

    def __init__(self, nClusters=None, randomState=None, dataFrame=None):
        super().__init__(nClusters, dataFrame=dataFrame)
        self.algorithmObject = AgglomerativeClustering(n_clusters=self.nClusters) # linkage is the distance between clusters.
        self.name = "Agglomerative"

    # Can't change the sklearn Agglomerative random state
    def checkAgainstExternalClass(self,randomStateList, externalClass):
        return super().checkAgainstExternalClass([42], externalClass)

    def getSilhouetteScoreList(self, randomStateList):
        return len(randomStateList) * super().getSilhouetteScoreList([42]) # list with the same value.
