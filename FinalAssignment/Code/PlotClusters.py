from GlobalFunctions import get_folder_path
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import GlobalParameters
import ClusteringAlgorithmsImportFile
from Dataset1 import Dataset1
from Dataset2 import Dataset2
from GlobalFunctions import get_results_folder_path, get_dataset_folder_name, get_folder_path, get_file_path
import GlobalFunctions

class PlotClusters():

    def get_plot_file_name(self, random_state):
        self.file_name= f"ClusteringPlotWithRandomState{random_state}.png"
        return self.file_name
    
    def get_plot_file_path(self, random_state, dataset_index):
        self.get_plot_file_name(random_state)
        return GlobalFunctions.get_plot_file_path(self.file_name, dataset_index, GlobalParameters.CLUSTERING_PLOT_FOLDER_NAME)

    def __init__(self, clusteringAlgorithmList: list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list):
        """

        Args:
            clusteringAlgorithmList (list, optional): Plot the data using these clustering algorithms. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.clusteringAlgorithms = clusteringAlgorithmList
        self.is_remove_outliers = True
        self.is_DBSCAN = True

    def plotAndSaveOne(self, dataSet):
        """
        Save a fig of the clustering result for each clustering algorithm in our list.

        Args:
            dataSet (DataSet object): the data set we want to plot the clusters of.
        """
        dataSet.prepareDataset()

        datasetIndex = dataSet.get_index()
        data = dataSet.get_data_frame()
        algoNameSilhouetteScoreDict = {}
        clean_data_algo_name_sil_score_dict = {}
        randomState = GlobalParameters.random_state
        nrows = 2
        ncols = 3
        fontsize = 20
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(23, 10))
        fig.suptitle(
            f"Data-Set {datasetIndex} Clustering With Optimal \n Clusters Number And Random State {randomState}")
        # axes indexes
        i = 0
        j = 0

        for clusterAlgo in self.clusteringAlgorithms[0:3]:
            clusterAlgo.setDataFrame(data)
            # clusterAlgo.setNClustersDatasetIndex(datasetIndex)
            clusterAlgo.setNClusters(dataSet.get_n_clusters())
            algoNameSilhouetteScoreDict[clusterAlgo.name] = clusterAlgo.getSilhouetteScoreList(
                [randomState])[0]  # function gets a list and returns a list.
            labels = clusterAlgo.getLabels()
            
            ax[i, j].scatter(data['dim1'], data['dim2'], c=labels)
            if self.is_remove_outliers:
                anomalous_points_df, regular_points_df = clusterAlgo.get_anomalous_dataframe_negative_silhouette_coefficients(dataSet)
                clusterAlgo.setDataFrame(regular_points_df)

                clean_data_algo_name_sil_score_dict[clusterAlgo.name] = clusterAlgo.getSilhouetteScoreList(
                [randomState])[0]
                
                ax[i, j].scatter(anomalous_points_df['dim1'], anomalous_points_df['dim2'], c='black')
            
                clean_data_algo_name_sil_score_dict[clusterAlgo.name] = 0.3

            ax[i, j].set_title(
                f"{clusterAlgo.name} With {clusterAlgo.getNClusters()} Clusters", fontsize=fontsize)

            j += 1
            if j == ncols:
                j = 0
                i += 1

        # ---------- bar plot for silhouette score ----------
        labels = algoNameSilhouetteScoreDict.keys()
        x = np.arange(len(labels))
        width = 0.3
        if self.is_remove_outliers:
            ax[i, j].bar(x - width/2, algoNameSilhouetteScoreDict.values(), width=width, color='black', label='All Data')
            ax[i, j].bar(x + width/2, clean_data_algo_name_sil_score_dict.values(), width=width, label = 'Data With Anomalous Points Removed')
            ax[i, j].set_xticks(x)
            ax[i, j].set_xticklabels(labels)
            ax[i, j].legend()
        else:
            ax[i, j].bar(x, algoNameSilhouetteScoreDict.values())
        ax[i, j].set_title(f"Silhouette Score", fontsize=fontsize)
        

        # ---------- Save Plot ----------
        plot_file_path = self.get_plot_file_path(randomState, datasetIndex)
        plt.savefig(plot_file_path)
        plt.close()

# Due to some memory interference issues with saving the figure and acessing the CSV file for loading the data, this function can't loop over the datasets and save their figures in one run.
if __name__ == '__main__':
    plotClusters = PlotClusters()
    # plotClusters.plotAndSaveOne(Dataset1())
    plotClusters.plotAndSaveOne(Dataset2())
