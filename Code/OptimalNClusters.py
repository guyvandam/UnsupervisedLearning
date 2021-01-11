import os
from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ClusteringAlgorithms
import DataSets


class OptimalNClusters:
    def __init__(self, clusteringAlgorithmList: list = ClusteringAlgorithms.clusteringAlgorithmList):
        """
        init method.

        Args:
            clusteringAlgorithmList (list, optional): list of ClusteringAlgorithm objects for us to get the optimal NClusters of. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.clusteringAlgorithmList = clusteringAlgorithmList

    def runRandomStates(self, dataset, maxNClusters: int, randomStateList: list):
        """
        Get the optimal number of clusters for each algorithm for every random state in the random state list, given the input data-set.
        Saves the results in a CSV file.

        Args:
            dataset (DataSet object): the dataset we want to get the optimal number of clusters for.
            maxNClusters (int): maximum number of clusters to check.
            randomStateList (list): list of different Random states
        """
        randomStateNClustersDict = {}
        for randomState in randomStateList:
            randomStateNClustersDict[str(randomState)] = self.optimalNClusters(
                dataset, maxNClusters, randomState)

        resultDf = pd.DataFrame(randomStateNClustersDict)
        maxFrequencyColumn = []
        averageColumn = []
        for _, row in resultDf.iterrows():
            maxFrequencyColumn.append(mode(row))
            averageColumn.append(np.mean(row))

        resultDf['MostFrequent'] = maxFrequencyColumn
        resultDf['Average'] = averageColumn

        # ---------- Save results in a CSV file ----------
        directory = os.path.join(
            os.getcwd(), f"Results\\Dataset{dataset.getDatasetIndex()}\\OptimalNClusters")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        resultDf.to_csv(
            directory + f"\\{maxNClusters}ClusterRange{len(randomStateList)}RandomStates.csv")

    def optimalNClusters(self, dataset, maxNClusters: int, randomState: int) -> dict:
        """
        Calculate the optimal number of clusters for the dataset, with the input random state for each algorithm by taking the NClusters with the highest Silhouette score.
        Saves the Silhouette score plot.
        Returns a dict with the algorithms name and optimal NClusters.

        Args:
            dataset (DataSet object): the dataset we want to get the optimal number of clusters for.
            maxNClusters (int): maximum number of clusters to check.
            randomState (int): integer representing a random state.

        Returns:
            dict: key - algorithm name, value - optimal NClusters.
        """

        # key - algorithm name, value - list of silhouette scores for each nClusters
        algoNameSillScoreDict = {}
        algoNameMaxScoreDict = {}
        nClustersRange = range(2, maxNClusters + 1)
        for clusterAlgo in self.clusteringAlgorithmList:
            sillScoreList = []
            clusterAlgo.setDataFrame(dataset.getDataFrame())
            for nClusters in nClustersRange:
                print(
                    f"{clusterAlgo.getName()} Clustering dataset {dataset.getDatasetIndex()} with {nClusters} Clusters and Random state {randomState}")
                clusterAlgo.setNClusters(nClusters)
                clusterAlgo.createLabels()
                sillScore = clusterAlgo.getSilhouetteScore()
                sillScoreList.append(sillScore)

            algoNameSillScoreDict[clusterAlgo.getName()] = sillScoreList

        for name, sillScoreList in algoNameSillScoreDict.items():
            plt.plot(nClustersRange, sillScoreList, 'o-', label=name)
            algoNameMaxScoreDict[name] = nClustersRange[np.argmax(
                sillScoreList)]
        plt.legend()
        plt.title(
            f"Silhouette Score For Data-Set {dataset.getDatasetIndex()} With Random State {randomState}")
        plt.xlabel("Number Of Clusters")
        plt.ylabel("Silhouette Score")

        # ---------- Save Plot ----------
        directory = os.path.join(
            os.getcwd(), f"Results\\Dataset{dataset.getDatasetIndex()}\\OptimalNClusters")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        plt.savefig(directory + f"\\RandomState{randomState}.png")
        plt.close()
        return algoNameMaxScoreDict


# 10 most comman random seeds.
randomStateList = [0, 1, 42, 1234, 10, 123, 2, 5, 12, 12345]

onc = OptimalNClusters()
for ds in DataSets.dataSetList:
    ds.prepareDataset()
    onc.runRandomStates(ds, 10, randomStateList)
