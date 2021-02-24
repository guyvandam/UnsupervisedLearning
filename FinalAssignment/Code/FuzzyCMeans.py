import numpy as np
import skfuzzy as fuzz

from ClusteringAlgorithm import ClusteringAlgorithm


class FuzzyCMeansAlgorithm(ClusteringAlgorithm):
    def __init__(self, nClusters: int = None, randomState: int = None, dataFrame=None):
        """
        init method.

        Args:
            nClusters (int, optional): number of clusters. Defaults to None.
            randomState (int, optional): random state. Defaults to None.
            dataFrame (pandas.DataFrame, optional): data to be clustered. Defaults to None.
        """
        super().__init__(nClusters, randomState, dataFrame=dataFrame)
        self.name = "FuzzyCMeans"

    def createLabels(self):
        """
        Uses the cmeans() method from skfuzzy, with default values to create the clustering labels.
        """
        # u is the probability for each point to be in the cluster
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(np.array(self.dataFrame).T, self.nClusters, m=2, error=0.005,
                                                  maxiter=1000, seed=self.randomState)
        self.labels = np.argmax(u, axis=0)
