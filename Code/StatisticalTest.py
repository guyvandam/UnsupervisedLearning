from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import numpy as np
import pandas as pd
import os
import GlobalParameters
import DataSets
import ClusteringAlgorithms


class StatisticalTest():
    def __init__(self, clusteringAlgorithmList=None, randomStateList=None):
        self.clusteringAlgorithmList = ClusteringAlgorithms.clusteringAlgorithmsList if clusteringAlgorithmList is None else clusteringAlgorithmList
        self.randomStateList = GlobalParameters.randomStates if randomStateList is None else randomStateList

        self.result = {}

    def run(self, dataset):
        dataset.prepareDataset()

        for clusterAlgorithm in self.clusteringAlgorithmList:
            clusterAlgorithm.setDataFrame(dataset.getDataFrame())
            clusterAlgorithm.setNumClustersDatasetIndex(
                dataset.getDatasetIndex())

        winner = self.clusteringAlgorithmList[0]
        winnerSilhouetteList = winner.getSilhouetteScoreList(self.randomStateList)
        winnerAvg = np.mean(winnerSilhouetteList)

        for candidate in self.clusteringAlgorithmList[1:]:
            candidateSilhouetteList = candidate.getSilhouetteScoreList(self.randomStateList)
            candidateAvg = np.mean(candidateSilhouetteList)

            # checking if candidate mean is bigger than winner mean.
            # H0 = meanW >= meanC
            stat, pValue = ttest_ind(candidateSilhouetteList, winnerSilhouetteList, equal_var=False)
            if stat < 0:
                realPValue = 1 - pValue / 2
            else:
                realPValue = pValue / 2

            columnKey = f"{winner.name}Against{candidate.name}"
            if realPValue < 0.05:
                winner = candidate
                winnerSilhouetteList = candidateSilhouetteList
                winnerAvg = candidateAvg

            columnValue = {"pValue": realPValue, "Stat": stat, "Cluster1Avg": winnerAvg, "Cluster2Avg": candidateAvg,
                           "Winner": winner.getName()}

            self.result[columnKey] = columnValue

        self.result = pd.DataFrame(self.result)
        print(self.result)
        directory = os.path.join(os.getcwd(), f"Results\\Dataset{dataset.getDatasetIndex()}\\StatisticalTest")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        self.result.to_csv(directory + f"\\StatisticalTestWith{len(self.randomStateList)}RandomStates.csv")


for ds in DataSets.dataSetList:
    ST = StatisticalTest()
    ST.run(ds)