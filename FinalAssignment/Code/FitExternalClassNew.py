import os

import matplotlib.pyplot as plt
import pandas as pd

import ClusteringAlgorithmsImportFile
from Dataset1 import Dataset1
from Dataset2 import Dataset2
import GlobalParameters
import GlobalFunctions
from StatisticalTestC import sort_df_by_stat_test

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
        self.clusteringAlgorithmList = clusteringAlgorithmList[0:2]
        self.randomStateList = randomStateList

    def get_csv_file_name(self):
        file_name = f"{self.n_clusters}ClustersAnd{len(self.randomStateList)}RandomStates.csv"
        return file_name
    
    def get_csv_file_path(self, dataset_index):
        file_name = self.get_csv_file_name()
        return get_external_fittment_file_path(dataset_index, file_name)

    def get_bar_chart_file_name(self):
        file_name = f"{self.n_clusters}ClustersAnd{len(self.randomStateList)}RandomStates.png"
        return file_name
    
    def get_bar_chart_fie_path(self, dataset_index):
        file_name = self.get_bar_chart_file_name()
        return get_external_fittment_file_path(dataset_index,file_name)

    def createCSV(self, dataset):
        """
        For each of the external classifier labels and each of the clustering algorithms we save the fitment score in a dictionary using the checkAgainstExternalClassRandomStateList() mehtod of the ClusteringAlgorithm object.
        This dictionary later transforms into a dataframe that can easily be saved in a CSV file, using pandas to_csv() mehtod.
        We also save a plot of the results.

        Args:
            dataSet (DataSet object): data-set we want to check the fitment between external labels and prediction labels.
        """
        dataset.prepareDataset()
        self.n_clusters = dataset.get_n_clusters()
        print(f"Checking External Classifier With {self.n_clusters} Clusters")
        result = {}
        dataset_index = dataset.get_index()
        for clusteringAlgorithm in self.clusteringAlgorithmList:
            clusteringAlgorithm.setDataFrame(dataset.get_data_frame())
            print(clusteringAlgorithm.get_name())
            result[clusteringAlgorithm.get_name()] = clusteringAlgorithm.check_against_external_class_random_state_list(self.randomStateList, dataset.get_ground_truth())

        # ---------- Results into a DataFrame and add min and max columns ----------
        result_df = pd.DataFrame(result)
        print(result_df)

        stat_test_results_df, sorted_df = sort_df_by_stat_test(result_df)
        sorted_df.loc['mean'] = sorted_df.mean()

        ############################# save stat test results.
        file_name = f"{self.n_clusters}Clusters{len(self.randomStateList)}RandomStatesStatisticalTestResults.csv"
        csv_file_path = get_external_fittment_file_path(dataset_index, file_name)
        stat_test_results_df.to_csv(csv_file_path)

        ############################# save sorted df
        file_name = f"{self.n_clusters}Clusters{len(self.randomStateList)}RandomStates.csv"
        csv_file_path = get_external_fittment_file_path(dataset_index, file_name)
        sorted_df.to_csv(csv_file_path)


        # ---------- Save bar plot ----------
        algo_name_average_fit_dict = sorted_df.mean().to_dict()
        print(algo_name_average_fit_dict)
        fig, ax = plt.subplots(figsize=(10, 8))
        # algoNameAverageFitDict = dict(sorted(algoNameAverageFitDict.items(), key=lambda item: item[1]))

        ax.bar(list(algo_name_average_fit_dict.keys()),
                algo_name_average_fit_dict.values())
         
        fig.suptitle(
            # f"Sorted Average Of Kullback-Leibler Divergence Between \n Prediction Labels And External Classifier With {self.n_clusters} Clusters For \n Data-Set {dataset_index} Across {len(self.randomStateList)} Random States", fontsize = 17)            f"Sorted Average Of Kullback-Leibler Divergence Between \n Prediction Labels And External Classifier With {self.n_clusters} Clusters For \n Data-Set {dataset_index} Across {len(self.randomStateList)} Random States", fontsize = 17)
            f"Sorted Average Of Kullback-Leibler Divergence Between Prediction Labels And \n External Classifier With {self.n_clusters} Clusters For Data-Set {dataset_index} Across {len(self.randomStateList)} Random States", fontsize = 18)

        bar_chart_file_path = self.get_bar_chart_fie_path(dataset_index)
        fig.savefig(bar_chart_file_path)
        plt.close()


# Due to some memory interference issues with saving the figure and acessing the CSV file for loading the data, this function can't loop over the datasets and save their figures in one run.
if __name__ == '__main__':
    fec = FitExternalClass()
    fec.createCSV(Dataset1())
    # fec.createCSV(Dataset2())
