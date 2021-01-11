from sklearn.cluster import AgglomerativeClustering

from ClusteringAlgorithm import ClusteringAlgorithm


class AgglomerativeClusteringAlgorithm(ClusteringAlgorithm):

    def __init__(self, nClusters: int = None, dataFrame=None):
        """
        Initializing the AgglomerativeClustering object from sklearn.cluster, with the input number of clusters.
        Sets the algorithm name.
        Args:
            nClusters (int, optional): [description]. number of clusters.
            dataFrame (pandas.DataFrame, optional): [description]. data to be clustered.
        """
        super().__init__(nClusters, dataFrame=dataFrame)
        self.algorithmObject = AgglomerativeClustering(
            n_clusters=self.nClusters)
        self.name = "Agglomerative"

    def checkAgainstExternalClass(self, randomStateList, externalLabels) -> list:
        """
        AgglomerativeClustering object from sklearn.cluster doesn't get a random state as input, so the is no need to
        call to parent function that repeats for every random state.
        """
        return super().checkAgainstExternalClass([42], externalLabels)

    def getSilhouetteScoreList(self, randomStateList):
        """
        AgglomerativeClustering object from sklearn.cluster doesn't get a random state as input, so the is no need to
        call to parent function that repeats for every random state.
        """
        return len(randomStateList) * super().getSilhouetteScoreList([42])
