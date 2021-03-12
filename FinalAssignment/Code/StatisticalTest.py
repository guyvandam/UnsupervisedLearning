import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import ClusteringAlgorithmsImportFile
import DatasetsImportFile
import GlobalParameters
import GlobalFunctions

def get_csv_file_name(num_random_stats):
    file_name = f"StatisticalTestWith{num_random_stats}RandomStates.csv"
    return file_name

def get_csv_file_path(num_random_states, dataset_index):
    file_name = get_csv_file_name(num_random_states)
    return GlobalFunctions.get_plot_file_path(file_name, dataset_index, GlobalParameters.STATISTICAL_TEST_FOLDER_NAME)

class StatisticalTest():
    def __init__(self, randomStateList: list = GlobalParameters.randomStateList,
                 clusteringAlgorithmList: list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list):
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
        dataset_index = dataSet.get_index()

        for clusterAlgorithm in self.clusteringAlgorithmList:
            clusterAlgorithm.setDataFrame(dataSet.get_data_frame())
            # clusterAlgorithm.setNClustersDatasetIndex(dataSet.getDatasetIndex()
            clusterAlgorithm.setNClusters(dataSet.get_n_classes())

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
        csv_file_path = get_csv_file_path(len(self.randomStateList), dataset_index)
        self.result.to_csv(csv_file_path)


for ds in DatasetsImportFile.dataset_obj_list:
    ST = StatisticalTest()
    ST.createCSV(ds)
