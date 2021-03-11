from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DatasetsImportFile
import ClusteringAlgorithmsImportFile
import GlobalParameters
from GlobalFunctions import get_results_folder_path, get_dataset_folder_name, get_folder_path, get_file_path

def get_plot_file_name(random_state):
    file_name = f"RandomState{random_state}.png"
    return file_name

def get_csv_file_name(min_n_clusters, max_n_clusters, random_state_list_length):
    file_name = f"{min_n_clusters}-{max_n_clusters}ClusterRange{random_state_list_length}RandomStates.csv"
    return file_name

def get_optimal_n_clusters_file_path(file_name, dataset_index):
    

    results_folder_path = get_results_folder_path()
    optimal_n_clusters_folder_path = get_folder_path(folder_name = GlobalParameters.OPTIMAL_N_CLUSTERS_FOLDER_NAME, enclosing_path = results_folder_path)
    dataset_folder_name = get_dataset_folder_name(dataset_index)
    dataset_folder_path = get_folder_path(dataset_folder_name, enclosing_path = optimal_n_clusters_folder_path)

    plot_file_path = get_file_path(file_name, dataset_folder_path)
    return plot_file_path

class OptimalNClusters:
    def __init__(self, clusteringAlgorithmList: list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list):
        """
        init method.

        Args:
            clusteringAlgorithmList (list, optional): list of ClusteringAlgorithm objects for us to get the optimal NClusters of. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.clusteringAlgorithmList = clusteringAlgorithmList

    def runRandomStates(self, dataset, min_n_clusters: int, maxNClusters: int, randomStateList: list):
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
                dataset, min_n_clusters, maxNClusters, randomState)

        resultDf = pd.DataFrame(randomStateNClustersDict)
        maxFrequencyColumn = []
        averageColumn = []
        for _, row in resultDf.iterrows():
            maxFrequencyColumn.append(mode(row))
            averageColumn.append(np.mean(row))

        resultDf['MostFrequent'] = maxFrequencyColumn
        resultDf['Average'] = averageColumn

        # ---------- Save results in a CSV file ----------
        file_name = get_csv_file_name(min_n_clusters, maxNClusters, len(randomStateList))
        file_path = get_optimal_n_clusters_file_path(file_name, dataset.get_index())
        
        resultDf.to_csv(file_path)


    def optimalNClusters(self, dataset, minNClusters: int, maxNClusters: int, randomState: int) -> dict:
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
        nClustersRange = range(minNClusters, maxNClusters + 1)
        dataset_index = dataset.get_index()
        
        for clusterAlgo in self.clusteringAlgorithmList:
            sillScoreList = []
            clusterAlgo.setDataFrame(dataset.get_data_frame())
            for nClusters in nClustersRange:
                print(f"{clusterAlgo.getName()} Clustering dataset {dataset_index} with {nClusters} Clusters and Random state {randomState}")
                clusterAlgo.setNClusters(nClusters)
                clusterAlgo.createLabels()
                sillScore = clusterAlgo.getSilhouetteScore()
                sillScoreList.append(sillScore)

            algoNameSillScoreDict[clusterAlgo.getName()] = sillScoreList

        for name, sillScoreList in algoNameSillScoreDict.items():
            plt.plot(nClustersRange, sillScoreList, 'o-', label=name)
            algoNameMaxScoreDict[name] = nClustersRange[np.argmax(sillScoreList)]
        
        ############################## plotting.
        plt.legend()
        plt.title(f"Silhouette Score For Data-Set {dataset_index} With Random State {randomState}")
        plt.xlabel("Number Of Clusters")
        plt.ylabel("Silhouette Score")

        # ---------- Save Plot ----------
        file_name = get_plot_file_name(random_state=randomState)
        file_path = get_optimal_n_clusters_file_path(file_name=file_name, dataset_index=dataset_index)
        
        plt.savefig(file_path)
        plt.close()
        return algoNameMaxScoreDict

if __name__ == "__main__":
    # 10 most comman random seeds.
    randomStateList = [0, 1, 42, 1234, 10, 123, 2, 5, 12, 12345]

    onc = OptimalNClusters()

    ################### try 3 more clusters.
    num_n_clusters_tries = 3
    for ds in DatasetsImportFile.dataset_obj_list[1:2]:
        ds.prepareDataset()
        num_classes = ds.get_n_classes()
        onc.runRandomStates(ds, num_classes, (num_classes + num_n_clusters_tries), randomStateList)
