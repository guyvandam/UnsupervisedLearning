from sklearn.cluster import KMeans

from ClusteringAlgorithmInterface import ClusteringAlgorithm


class KMeansAlgorithm(ClusteringAlgorithm):
    def __init__(self, nClusters: int = None, randomState: int = None, dataFrame=None):
        """
        Initializing the KMeans object from sklearn.cluster, with the input number of clusters.
        Sets the algorithm name.

        Args:
            nClusters (int, optional): number of clusters. Defaults to None.
            randomState (int, optional): random state. Defaults to None.
            dataFrame (pandas.DataFrame, optional): data to be clustered. Defaults to None.
        """
        super().__init__(nClusters, randomState, dataFrame=dataFrame)
        self.algorithm_object = KMeans(
            n_clusters=self.nClusters, random_state=self.randomState)
        self.name = "KMeans"
