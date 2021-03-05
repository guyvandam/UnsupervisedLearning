from sklearn.mixture import GaussianMixture

from ClusteringAlgorithm import ClusteringAlgorithm


class GMMAlgorithm(ClusteringAlgorithm):

    def __init__(self, nClusters: int = None, randomState: int = None, dataFrame=None):
        """
        Initializing the GaussianMixture object from sklearn.mixture, with the input number of clusters.
        Sets the algorithm name.

        Args:
            nClusters (int, optional): number of clusters. Defaults to None.
            randomState (int, optional): random state. Defaults to None.
            dataFrame (pandas.DataFrame, optional): data to be clustered. Defaults to None.
        """
        super().__init__(nClusters, randomState, dataFrame=dataFrame)
        self.algorithm_object = GaussianMixture(n_components=self.nClusters, random_state=self.randomState)
        self.name = "GMM"
