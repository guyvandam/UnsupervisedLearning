import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import ClusteringAlgorithms
import DataSets
import GlobalParameters


class StatisticalTest():
    def __init__(self, randomStateList: list = GlobalParameters.randomStateList,
                 clusteringAlgorithmList: list = ClusteringAlgorithms.clusteringAlgorithmList):
        """
        init method.

        Args:
            randomStateList (list, optional): list of random states. Defaults to GlobalParameters.randomStateList.
            clusteringAlgorithmList (list, optional): list of Clustering Algorithm Objects. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.clusteringAlgorithmList = clusteringAlgorithmList
        self.randomStateList = randomStateList

        self.result = {}

    def createCSV(self, dataSet):
        """
        We perfrom the statistical test seeing in the paper. essentially finding the maximum with our newley defined order.
        We save the result in a dictionary which later transforms into a pandas.DataFrame which is saved to a CSV file.

        Args:
            dataSet (DataSet object): data-set we want to check the fitment between external labels and prediction labels.
        """
        dataSet.prepareDataset()

        for clusterAlgorithm in self.clusteringAlgorithmList:
            clusterAlgorithm.setDataFrame(dataSet.getDataFrame())
            clusterAlgorithm.setNClustersDatasetIndex(
                dataSet.getDatasetIndex())

        winner = self.clusteringAlgorithmList[0]
        winnerSilhouetteList = winner.getSilhouetteScoreList(
            self.randomStateList)
        winnerAvg = np.mean(winnerSilhouetteList)

        for candidate in self.clusteringAlgorithmList[1:]:
            candidateSilhouetteList = candidate.getSilhouetteScoreList(
                self.randomStateList)
            candidateAvg = np.mean(candidateSilhouetteList)

            # checking if candidate mean is bigger than winner mean.
            # H0 = meanW >= meanC, we need to send test_ind(candidate, winner)
            stat, pValue = ttest_ind(
                candidateSilhouetteList, winnerSilhouetteList, equal_var=False)
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
        directory = os.path.join(
            os.getcwd(), f"Results\\Dataset{dataSet.getDatasetIndex()}\\StatisticalTest")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        self.result.to_csv(
            directory + f"\\StatisticalTestWith{len(self.randomStateList)}RandomStates.csv")


for ds in DataSets.dataSetList:
    ST = StatisticalTest()
    ST.createCSV(ds)
