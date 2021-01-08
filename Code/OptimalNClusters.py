import os
from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ClusteringAlgorithms
import DataSets


class OptimalNClusters():
    # def __init__(self, maxNCluster, clusteringAlgorithmList, randomState):
    def __init__(self, clusteringAlgorithmList=None):
        self.clusteringAlgorithmList = clusteringAlgorithmList
        
        if self.clusteringAlgorithmList is None: self.clusteringAlgorithmList = ClusteringAlgorithms.clusteringAlgorithmsList

    def runRandomStates(self, dataset, maxNClusters, randomStateList):
        randomStateNClustersDict = {}
        for randomState in randomStateList:
            randomStateNClustersDict[str(randomState)] = self.optimalNClusters(dataset, maxNClusters, randomState)

        resultDf = pd.DataFrame(randomStateNClustersDict)
        maxFrequencyColumn = []
        averageColumn = []
        for _, row in resultDf.iterrows():
            maxFrequencyColumn.append(mode(row))
            averageColumn.append(np.mean(row))
        
        resultDf['MostFrequent'] = maxFrequencyColumn
        resultDf['Average'] = averageColumn

        # ---------- Save results in a CSV file ----------
        directory = os.path.join(os.getcwd(), f"Results\\Dataset{dataset.getDatasetIndex()}\\OptimalNClusters")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        resultDf.to_csv(directory+f"\\{maxNClusters}ClusterRange{len(randomStateList)}RandomStates.csv")


    def optimalNClusters(self, dataset, maxNClusters, randomState):
        # key - algorithm name, value - list of silhouette scores for each nClusters
        algoNameSillScoreDict = {}
        algoNameMaxScoreDict = {}
        nClustersRange = range(2, maxNClusters + 1)
        for clusterAlgo in self.clusteringAlgorithmList:
            sillScoreList = []
            clusterAlgo.setDataFrame(dataset.getDataFrame())
            for nClusters in nClustersRange:
                print(f"{clusterAlgo.getName()} Clustering dataset {dataset.getDatasetIndex()} with {nClusters} Clusters and Random state {randomState}")
                clusterAlgo.setNClusters(nClusters)
                sillScore = clusterAlgo.getSilhouetteScore()
                sillScoreList.append(sillScore)

            algoNameSillScoreDict[clusterAlgo.getName()] = sillScoreList

        for name, sillScoreList in algoNameSillScoreDict.items():
            plt.plot(nClustersRange, sillScoreList, 'o-', label=name)
            algoNameMaxScoreDict[name] = nClustersRange[np.argmax(sillScoreList)]
        plt.legend()

        # ---------- Save Plot ----------
        directory = os.path.join(os.getcwd(), f"Results\\Dataset{dataset.getDatasetIndex()}\\OptimalNClusters")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        plt.savefig(directory + f"\\RandomState{randomState}.png")
        plt.close()
        return algoNameMaxScoreDict


randomStateList = [0, 1, 42, 1234, 10, 123, 2, 5, 12, 12345] # 10 most comman random seeds.


onc = OptimalNClusters()
for ds in DataSets.dataSetList:
    ds.prepareDataset()
    onc.runRandomStates(ds, 10, randomStateList)
