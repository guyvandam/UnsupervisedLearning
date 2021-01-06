from ClusterAlgorithm import ClusterAlgorithm
import pandas as pd
from LoadDataSet1 import LoadDataSet1
from LoadDataSet3 import LoadDataSet3
from LoadDataSet2 import LoadDataSet2
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from KMeans import KMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm


class FindNumOfClusters():

    def __init__(self, clusteringAlgorithms: list, loadData, numClustersRange: int):
        self.clusteringAlgorithms = clusteringAlgorithms
        self.loadData = loadData
        self.numClustersRange = numClustersRange
        self.numOfRandomStates = [0, 1, 42, 1234, 10, 123, 2, 5, 12, 12345] # 8 most comman random seeds.
        # self.numOfRandomStates = [0, 1, 42, 1234]

    def find(self):
        result = {}

        for randomState in self.numOfRandomStates:
            randomStateDict = {}
            for clusteringAlgorithm in self.clusteringAlgorithms:
                resultTemp = clusteringAlgorithm.getBestNumClusters(
                    randomState, self.numClustersRange, self.loadData.getDatasetIndex())
                randomStateDict[clusteringAlgorithm.name] = resultTemp
            result[randomState] = randomStateDict

        result = pd.DataFrame(result)
        print(result)
        result.to_csv(
            f"numOfClustersDataset{self.loadData.getDatasetIndex()}.csv")


    # def findWithExternal(self):
    #     result = {}

    #     for randomState in self.numOfRandomStates:
    #         randomStateDict = {}
    #         for clusteringAlgorithm in self.clusteringAlgorithms:
    #             resultTemp = clusteringAlgorithm.getBestNumClustersExternalClass(
    #                 randomState, self.numClustersRange, self.loadData)
    #             randomStateDict[clusteringAlgorithm.name] = resultTemp
    #         result[randomState] = randomStateDict

    #     result = pd.DataFrame(result)
    #     print(result)
    #     result.to_csv(
    #         f"numOfClustersDataset{self.loadData.getDatasetIndex()}ExternalLabels.csv")



    
loadDataList = [LoadDataSet3(),LoadDataSet2(),LoadDataSet1()]
# ld = LoadDataSet1()

for ld in loadDataList:
    ld.prepareDataset()
    print(ld.getDatasetIndex())
    clusteringAlgorithms = [
        KMeansAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        GMMAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        FuzzyCMeansAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        AgglomerativeClusteringAlgorithm(
            nComponents=1, dataFrame=ld.getDataFrame()),
        SpectralClusteringAlgorithm(nComponents=1, dataFrame=ld.getDataFrame())
    ]

    fn = FindNumOfClusters(clusteringAlgorithms, ld, 8)
    fn.find()
    # fn.findWithExternal()
