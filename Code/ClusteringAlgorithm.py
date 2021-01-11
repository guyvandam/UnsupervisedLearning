import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import silhouette_score


def getProbabilities(lst: list):
    # how many times each element appears.
    frequencyList = list(map(Counter(lst).get, lst))
    return np.array(frequencyList) / len(lst)


class ClusteringAlgorithm:
    def __init__(self, nClusters: int = None, randomState: int = None, dataFrame=None):
        """
        init function.

        Args:
            nClusters (int, optional): number of clusters. Defaults to None.
            randomState (int, optional): random state for initialization. Defaults to None.
            dataFrame (pandas.DataFrame, optional): data to be clustered. Defaults to None.
        """
        self.dataFrame = dataFrame
        self.nClusters = nClusters
        self.labels = None
        self.algorithmObject = None
        self.randomState = randomState
        self.name = None
        # key - dataset index, value - number of clusters
        self.dataSetIndexOptimalNClustersDict = None

    def setDataFrame(self, dataFrame: pd.DataFrame):
        """

        Args:
            dataFrame (pd.DataFrame): new data frame to be clustered.
        """
        self.dataFrame = dataFrame

    def initDataSetIndexOptimalNClustersDict(self):
        """
        Parse the OptimalClustersNumber.csv file into the dataSetIndexOptimalNClustersDict dictionary
        """
        path = str(os.path.join(
            os.getcwd(), "Results\\OptimalClustersNumber.csv"))
        numOfClustersDF = pd.read_csv(path)
        nClustersList = list(numOfClustersDF[self.name])
        # self.dataSetIndexOptimalNClustersDict = dict(zip(range(1, len(nClustersList) + 1), nClustersList))
        self.dataSetIndexOptimalNClustersDict = {i: k for i, k in zip(
            range(1, len(nClustersList) + 1), nClustersList)}  # more readable

    def createLabels(self):
        """
        Creates the algorithm labels using the fit_predict method most of the inherited classes use.


        Raises:
            RuntimeError: Data is not initialzed
        """
        if self.dataFrame is None:
            raise RuntimeError("Data not initialzed")
        self.labels = self.algorithmObject.fit_predict(self.dataFrame)

    def getLabels(self) -> np.ndarray:
        """

        Returns:
            np.ndarray: algorithm prediction labels.
        """
        return self.labels

    def getNClusters(self) -> int:
        """

        Returns:
            int: number of clusters.
        """
        return self.nClusters

    def getName(self) -> str:
        """

        Returns:
            str: algorithm name.
        """
        return self.name

    def getKLDivergence(self, externalClassList: list) -> float:
        """
        calculates the K-L divergence between the algorithm's prediction labels and the input external labels.
        computes the probability list using the getProbabilities() function defied above.
        uses the entropy() function from scipy.stats.
        Args:
            externalClassList (list): External Classifier labels.

        Returns:
            float: the output KL divergence.
        """
        if self.labels is None:
            self.createLabels()

        # entropy gets list of probabilities
        return entropy(pk=getProbabilities(externalClassList), qk=getProbabilities(self.labels))

    def setNClustersDatasetIndex(self, datasetIndex: int):
        """
        sets the number of clusters based on the input dataset according to the optimal number of clusters found in the dataSetIndexOptimalNClustersDict dictionary
        Args:
            datasetIndex (int): dataset index.
        """
        if self.dataSetIndexOptimalNClustersDict is None:
            self.initDataSetIndexOptimalNClustersDict()
        self.__init__(
            self.dataSetIndexOptimalNClustersDict[datasetIndex], self.randomState, self.dataFrame)

    def setNClusters(self, nClusters: int):
        """
        sets the number of clusters based on the input number of clusters.

        Args:
            nClusters (int): new number clusters.
        """
        if nClusters is None:
            return
        self.__init__(nClusters, self.randomState, self.dataFrame)

    def setRandomState(self, randomState: int):
        """
        sets the random state to be the input random state.

        Args:
            randomState (int): new random state
        """
        if randomState is None:
            return
        self.__init__(nClusters=self.nClusters,
                      randomState=randomState, dataFrame=self.dataFrame)

    def checkAgainstExternalClass(self, randomStateList: list, externalLabels: list) -> dict:
        """
        calculate fitment to external classifier using K-L divergence, with a list of different random states.
        Args:
            randomStateList (list): list of random states
            externalLabels (list): list of external labels

        Returns:
            dict: key - random state, value - KL divergence with that random state.
        """
        randomStateKLDivergenceDict = {}  # key - randoom state, value mutual info score.
        nClusters = len(set(externalLabels))
        self.setNClusters(nClusters)
        for randomState in randomStateList:
            self.setRandomState(randomState)
            self.createLabels()
            randomStateKLDivergenceDict[str(
                randomState)] = self.getKLDivergence(externalLabels)

        randomStateKLDivergenceDict['Average'] = np.mean(
            list(randomStateKLDivergenceDict.values()))
        if len(set(randomStateKLDivergenceDict.values())) == 1 and not len(randomStateList) == 1:
            print(f"Random State Doesn't make a change for {self.name}")
        return randomStateKLDivergenceDict

    def getSilhouetteScore(self) -> float:
        """
        returns the Silhouette score for the current labels using the silhouette_score function from sklearn.metrics

        Returns:
            float: Silhouette score
        """
        return silhouette_score(self.dataFrame, self.labels)

    def getSilhouetteScoreList(self, randomStateList: list) -> list:
        """
        returns the a list of Silhouette score with each random state.
        Args:
            randomStateList (list): list of different random states

        Returns:
            list: list of silhouette scores.
        """
        silhouetteList = []
        for randomState in randomStateList:
            self.setRandomState(randomState)
            self.createLabels()
            silhouetteList.append(self.getSilhouetteScore())

        if len(set(silhouetteList)) == 1 and not len(randomStateList) == 1:
            print(f"Random State Doesn't make a change for {self.getName()}")
        return silhouetteList
