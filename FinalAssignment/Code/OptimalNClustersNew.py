from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import DatasetsImportFile
import ClusteringAlgorithmsImportFile
import GlobalParameters
from GlobalFunctions import get_results_folder_path, get_dataset_folder_name, get_folder_path, get_file_path, get_df_by_path
from ClusteringAlgorithmsImportFile import clustering_algorithm_obj_list
from scipy import stats
from StatisticalTestC import sort_df_by_stat_test

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
    def __init__(self, dataset, min_n_clusters, max_n_clusters, clusteringAlgorithmList: list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list):
        """
        init method.

        Args:
            clusteringAlgorithmList (list, optional): list of ClusteringAlgorithm objects for us to get the optimal NClusters of. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.dataset = dataset
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = max_n_clusters
        self.random_state_list = GlobalParameters.random_state_list[:5]
        self.clusteringAlgorithmList = clusteringAlgorithmList[:2]

        self.is_create_new = False
    def run_all(self):
        n_clusters_range = range(self.min_n_clusters, self.max_n_clusters + 1)

        dataset_index = self.dataset.get_index()
        num_random_states = len(self.random_state_list)

        for cluster_algo_obj in self.clusteringAlgorithmList:
            temp_df = self.run_n_clusters_on_cluster_algo(cluster_algo_obj)

            sorted_columns_list = list(temp_df.columns)
            plt.plot(n_clusters_range, sorted_columns_list, 'o-', label=cluster_algo_obj.get_name())

        ############################## plotting.
        plt.legend()
        plt.title(f"Sorted Order By Silhouette Score Mean Popluation ensured by \n Two-Sided T-Test Over {num_random_states} Random States For Data-Set {dataset_index}")
        plt.xlabel("Number Of Clusters")
        plt.ylabel("Place In Sorted Order")

        # ---------- Save Plot ----------
        file_name = f"{self.min_n_clusters}-{self.max_n_clusters}ClusterRange{num_random_states}RandomStatesPlot"
        file_path = get_optimal_n_clusters_file_path(file_name=file_name, dataset_index=dataset_index)
        
        plt.savefig(file_path)
        plt.close()
        
    def get_cluster_algo_csv_file_path(self, algo_name):
        file_name = f"{self.min_n_clusters}-{self.max_n_clusters}ClusterRange{len(self.random_state_list)}RandomStates{algo_name}.csv"
        csv_file_path = get_optimal_n_clusters_file_path(file_name, self.dataset.get_index())
        return csv_file_path
    
    def run_n_clusters_on_cluster_algo(self, cluster_algo_obj):
        csv_file_path = self.get_cluster_algo_csv_file_path(cluster_algo_obj.get_name())
        if os.path.exists(csv_file_path) and self.is_create_new:
            random_state_sil_score_df = get_df_by_path(csv_file_path)
        else:
            random_state_sil_score_df = pd.DataFrame()

            for n_classes in range(self.min_n_clusters, self.max_n_clusters + 1):
                temp_df = self.get_silhouetter_score_df(cluster_algo_obj, n_classes)
                
                if random_state_sil_score_df.empty:
                    random_state_sil_score_df = temp_df
                else:
                    random_state_sil_score_df = pd.concat([random_state_sil_score_df, temp_df], axis = 1)

        print(f"{cluster_algo_obj.get_name()} \n {random_state_sil_score_df}")

        stat_test_results_df, sorted_df = sort_df_by_stat_test(random_state_sil_score_df)

        ##################################### save results
        csv_file_path = self.get_cluster_algo_csv_file_path(cluster_algo_obj.get_name())
        sorted_df.to_csv(csv_file_path)

        csv_file_path = self.get_cluster_algo_csv_file_path(cluster_algo_obj.get_name()+"StatisticalTestResults")
        stat_test_results_df.to_csv(csv_file_path)
        return sorted_df

    def get_silhouetter_score_df(self, clusterAlgo, n_clusters):
        dataset_df = self.dataset.get_data_frame()
        clusterAlgo.setNClusters(n_clusters)
        clusterAlgo.setDataFrame(dataset_df)

        silhouette_score_list = []


        for random_state in self.random_state_list:
            print(clusterAlgo.get_name(), "with", n_clusters, "clusters and random state", random_state)
            clusterAlgo.setRandomState(random_state)
            clusterAlgo.createLabels()
            temp_sil_score = clusterAlgo.getSilhouetteScore()
            silhouette_score_list.append(temp_sil_score)

        result_df = pd.DataFrame(silhouette_score_list, index=self.random_state_list)

        result_df.rename(columns = {0 : n_clusters}, inplace = True)
        
        return result_df        

    def run_anomaly(self):
        n_clusters = self.dataset.get_n_clusters()
        result_df = pd.DataFrame()
        for cluster_algo_obj in self.clusteringAlgorithmList:
            csv_file_path = self.get_cluster_algo_csv_file_path(cluster_algo_obj.get_name())
            if os.path.isfile(csv_file_path):
                print("reading from memory")
                all_data_sil_score_df = pd.read_csv(csv_file_path)[str(n_clusters)]
            else:
                all_data_sil_score_df = self.get_silhouetter_score_df(cluster_algo_obj, n_clusters)

            _, clean_data_df = cluster_algo_obj.get_anomalous_dataframe_negative_silhouette_coefficients(self.dataset)
            self.dataset.set_dataframe(clean_data_df)

            clean_data_sil_score_df = self.get_silhouetter_score_df(cluster_algo_obj, n_clusters)

            temp_diff_sil_score_df = clean_data_sil_score_df - all_data_sil_score_df
            print(cluster_algo_obj.get_name(), temp_diff_sil_score_df)
            if result_df.empty:
                result_df = temp_diff_sil_score_df
            else:
                result_df = pd.concat([result_df, temp_diff_sil_score_df], axis = 1)
        print(result_df)

if __name__ == "__main__":

    ################### try 3 more clusters.
    num_n_clusters_tries = 1
    for ds in DatasetsImportFile.dataset_obj_list[1:2]:
        ds.prepareDataset()
        num_classes = ds.get_n_clusters()
        
        onc = OptimalNClusters(ds, num_classes, num_classes + num_n_clusters_tries)
        
        # onc.run_stat_test(clustering_algorithm_obj_list[0], num_classes+1)
        # onc.run_n_clusters_on_cluster_algo(clustering_algorithm_obj_list[0])
        onc.run_all()
        # onc.run_anomaly()
        # onc.runRandomStates(ds, num_classes, (num_classes + num_n_clusters_tries), random_state_list[0:3])
