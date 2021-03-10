from sklearn.cluster import SpectralClustering
from ClusteringAlgorithmInterface import ClusteringAlgorithm


class SpectralClusteringAlgorithm(ClusteringAlgorithm):

    def __init__(self, nClusters: int = None, randomState: int = None, dataFrame=None):
        """
        Initializing the SpectralClustering object from sklearn.cluster, with the input number of clusters.
        Sets the algorithm name.

        Args:
            nClusters (int, optional): number of clusters. Defaults to None.
            randomState (int, optional): random state. Defaults to None.
            dataFrame (pandas.DataFrame, optional): data to be clustered. Defaults to None.
        """
        super().__init__(nClusters=nClusters, randomState=randomState, dataFrame=dataFrame)
        self.algorithm_object = SpectralClustering(
            n_clusters=self.nClusters, random_state=self.randomState)
        self.name = "Spectral"
