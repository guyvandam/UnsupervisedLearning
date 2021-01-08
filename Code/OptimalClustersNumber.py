import pandas as pd
from DataSet1 import DataSet1
from DataSet3 import DataSet3
from DataSet2 import DataSet2
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from KMeans import KMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm
from statistics import mode 
import os
import numpy as np

class FindNumOfClusters():

    def __init__(self, clusteringAlgorithms: list, loadData, numClustersRange: int):
        self.clusteringAlgorithms = clusteringAlgorithms
        self.loadData = loadData
        self.numClustersRange = numClustersRange
        self.randomStates = [0, 1, 42, 1234, 10, 123, 2, 5, 12, 12345] # 10 most comman random seeds.

    def find(self):
        result = {}

        for randomState in self.randomStates:
            randomStateDict = {}
            for clusteringAlgorithm in self.clusteringAlgorithms:
                resultTemp = clusteringAlgorithm.getBestNumClusters(
                    randomState, self.numClustersRange, self.loadData.getDatasetIndex())
                randomStateDict[clusteringAlgorithm.name] = resultTemp
            result[randomState] = randomStateDict

        result = pd.DataFrame(result)
        maxFrequencyColumn = []
        averageColumn = []
        for _, row in result.iterrows():
            maxFrequencyColumn.append(mode(row))
            averageColumn.append(np.mean(row))
        
        result['MostFrequent'] = maxFrequencyColumn
        result['Average'] = averageColumn

        # ---------- Save results in a CSV file ----------
        directory = os.path.join(os.getcwd(), f"Results\\Dataset{self.loadData.getDatasetIndex()}\\OptimalClustersNumber")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        result.to_csv(directory+f"\\{self.numClustersRange}ClusterRange{len(self.randomStates)}RandomStates.csv")


    
dataSetList = [DataSet3(), DataSet2(), DataSet1()]

for ld in dataSetList:
    ld.prepareDataset()
    print(ld.getDatasetIndex())
    clusteringAlgorithms = [
        KMeansAlgorithm(dataFrame=ld.getDataFrame()),
        GMMAlgorithm(dataFrame=ld.getDataFrame()),
        FuzzyCMeansAlgorithm(dataFrame=ld.getDataFrame()),
        AgglomerativeClusteringAlgorithm(dataFrame=ld.getDataFrame()),
        SpectralClusteringAlgorithm(dataFrame=ld.getDataFrame())
    ]

    fn = FindNumOfClusters(clusteringAlgorithms, ld, 10)
    fn.find()
