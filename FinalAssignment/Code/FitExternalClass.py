import os

import matplotlib.pyplot as plt
import pandas as pd

import ClusteringAlgorithmsImportFile
# from ClusteringAlgorithmsImportFile import clusteringAlgorithmsList
from Dataset1 import Dataset1
from Dataset2 import Dataset2
import GlobalParameters
import GlobalFunctions

def get_external_fittment_file_path(dataset_index, file_name):
    file_path = GlobalFunctions.get_plot_file_path(file_name, dataset_index, GlobalParameters.EXTERNAL_LABELS_FITTMENT_RESULTS_FOLDER)
    return file_path

class FitExternalClass:

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

    def get_csv_file_name(self):
        file_name = f"ExteranlClassifierWith{self.nClusters}ClustersAnd{len(self.randomStateList)}RandomStates.csv"
        return file_name
    
    def get_csv_file_path(self, dataset_index):
        file_name = self.get_csv_file_name()
        return get_external_fittment_file_path(dataset_index, file_name)


    def get_bar_chart_file_name(self):
        file_name = f"ExternalClassifierWith{self.nClusters}ClustersAnd{len(self.randomStateList)}RandomStates.png"
        return file_name
    
    def get_bar_chart_fie_path(self, dataset_index):
        file_name = self.get_bar_chart_file_name()
        return get_external_fittment_file_path(dataset_index,file_name)

    def createCSV(self, dataset):
        """
        For each of the external classifier labels and each of the clustering algorithms we save the fitment score in a dictionary using the checkAgainstExternalClass() mehtod of the ClusteringAlgorithm object.
        This dictionary later transforms into a dataframe that can easily be saved in a CSV file, using pandas to_csv() mehtod.
        We also save a plot of the results.

        Args:
            dataSet (DataSet object): data-set we want to check the fitment between external labels and prediction labels.
        """
        dataset.prepareDataset()
        self.nClusters = dataset.get_n_classes()
        print(f"Checking External Classifier With {self.nClusters} Clusters")
        result = {}
        dataset_index = dataset.get_index()
        for clusteringAlgorithm in self.clusteringAlgorithmList:
            clusteringAlgorithm.setDataFrame(dataset.get_data_frame())
            print(clusteringAlgorithm.getName())
            result[clusteringAlgorithm.getName()] = clusteringAlgorithm.checkAgainstExternalClass(self.randomStateList, dataset.get_ground_truth())

        # ---------- Results into a DataFrame and add min and max columns ----------
        resultDF = pd.DataFrame(result)
        maxList = resultDF.idxmax(axis=1)
        minList = resultDF.idxmin(axis=1)
        resultDF['Max'] = maxList
        resultDF['Min'] = minList

        # ---------- Save results in a CSV file ----------
        csv_file_path = self.get_csv_file_path(dataset_index)
        resultDF.to_csv(csv_file_path)

        # ---------- Save bar plot ----------
        algoNameAverageFitDict = {name: fitmentDict['Average'] for name, fitmentDict in result.items()}
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(list(algoNameAverageFitDict.keys()),
                algoNameAverageFitDict.values())
        fig.suptitle(
            f"Average Of Kullback-Leibler Divergence Between Prediction Labels And External \n External Classifier With {self.nClusters} Clusters For Data-Set {dataset_index} Across {len(self.randomStateList)} Random States")
        
        bar_chart_file_path = self.get_bar_chart_fie_path(dataset_index)
        fig.savefig(bar_chart_file_path)
        plt.close()


# Due to some memory interference issues with saving the figure and acessing the CSV file for loading the data, this function can't loop over the datasets and save their figures in one run.
if __name__ == '__main__':
    fec = FitExternalClass()
    fec.createCSV(Dataset1())
    # fec.createCSV(DataSet2())
