from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DatasetsImportFile
import ClusteringAlgorithmsImportFile
import GlobalParameters
from GlobalFunctions import get_results_folder_path, get_dataset_folder_name, get_folder_path, get_file_path
from GlobalParameters import random_state_list
from ClusteringAlgorithmsImportFile import clustering_algorithm_obj_list
from scipy import stats

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

def sample_and_popluation_mean_test(samples):
        pop_mean = np.mean(samples)
        
        # H0 - sample mean = population mean.
        p_value, stat = stats.ttest_1samp(samples, pop_mean)

        # print("p value:", p_value, "stat:", stat)
        return p_value, stat, pop_mean

class OptimalNClusters:
    def __init__(self, dataset, min_n_clusters, max_n_clusters, clusteringAlgorithmList: list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list):
        """
        init method.

        Args:
            clusteringAlgorithmList (list, optional): list of ClusteringAlgorithm objects for us to get the optimal NClusters of. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.dataset = dataset
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.random_state_list = GlobalParameters.random_state_list[:2]
        self.clusteringAlgorithmList = clusteringAlgorithmList[0:2]

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
        
        for clusterAlgo in self.clusteringAlgorithmList[0:1]:
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

    def run_all(self):
        # result_df = pd.DataFrame()
        # algo_name_sil_score_dict = {}
        n_clusters_range = range(self.min_n_clusters, self.max_n_clusters + 1)

        dataset_index = self.dataset.get_index()
        num_random_states = len(self.random_state_list)

        for cluster_algo_obj in self.clusteringAlgorithmList:
            temp_df = self.run_n_clusters_on_cluster_algo(cluster_algo_obj)

            mean_sil_score_list = list(temp_df.loc['Mean'])
            plt.plot(n_clusters_range, mean_sil_score_list, 'o-', label=cluster_algo_obj.getName())

        ############################## plotting.
        plt.legend()
        plt.title(f"Silhouette Score Average Across {num_random_states} Random States For Data-Set {dataset_index}")
        plt.xlabel("Number Of Clusters")
        plt.ylabel("Average Silhouette Score")

        # ---------- Save Plot ----------
        file_name = f"{self.min_n_clusters}-{self.max_n_clusters}ClusterRange{num_random_states}RandomStatesPlot"
        file_path = get_optimal_n_clusters_file_path(file_name=file_name, dataset_index=dataset_index)
        
        plt.savefig(file_path)
        plt.close()
        

    def run_n_clusters_on_cluster_algo(self, cluster_algo_obj):
        random_state_sil_score_df = pd.DataFrame()

        for n_classes in range(self.min_n_clusters, self.max_n_clusters + 1):
            temp_df = self.get_silhouetter_score_df(cluster_algo_obj, n_classes)

            if random_state_sil_score_df.empty:
                random_state_sil_score_df = temp_df
            else:
                random_state_sil_score_df = pd.concat([random_state_sil_score_df, temp_df], axis = 1)
        
        df_to_save = random_state_sil_score_df.copy()
        maxList = df_to_save.idxmax(axis=1)
        df_to_save['max_value_n_clusters'] = maxList

        ##################################### save results
        file_name = f"{self.min_n_clusters}-{self.max_n_clusters}ClusterRange{len(self.random_state_list)}RandomStates{cluster_algo_obj.getName()}.csv"
        csv_file_path = get_optimal_n_clusters_file_path(file_name, self.dataset.get_index())
        df_to_save.to_csv(csv_file_path)

        return random_state_sil_score_df

    def get_silhouetter_score_df(self, clusterAlgo, n_classes):
        dataset_df = self.dataset.get_data_frame()
        clusterAlgo.setNClusters(n_classes)
        clusterAlgo.setDataFrame(dataset_df)

        silhouette_score_list = []


        for random_state in self.random_state_list:
            print(clusterAlgo.getName(), "with", n_classes, "clusters and random state", random_state)
            clusterAlgo.setRandomState(random_state)
            clusterAlgo.createLabels()
            temp_sil_score = clusterAlgo.getSilhouetteScore()
            silhouette_score_list.append(temp_sil_score)

        result_df = pd.DataFrame(silhouette_score_list, index=self.random_state_list).T

        p_value, stat, mean = sample_and_popluation_mean_test(silhouette_score_list)
        result_df['Mean'] = mean
        result_df['P-Value'] = p_value            
        result_df['T-Statistics'] = stat

        result_df = result_df.T
        result_df.rename(columns = {0 : n_classes}, inplace = True)
        
        return result_df        
    
if __name__ == "__main__":

    ################### try 3 more clusters.
    num_n_clusters_tries = 1
    for ds in DatasetsImportFile.dataset_obj_list[1:2]:
        ds.prepareDataset()
        num_classes = ds.get_n_classes()
        
        onc = OptimalNClusters(ds, num_classes, num_classes + num_n_clusters_tries)
        
        # onc.run_stat_test(clustering_algorithm_obj_list[0], num_classes+1)
        # onc.run_n_clusters_on_cluster_algo(clustering_algorithm_obj_list[0])
        onc.run_all()
        # onc.runRandomStates(ds, num_classes, (num_classes + num_n_clusters_tries), random_state_list[0:3])
