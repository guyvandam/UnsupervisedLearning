from sklearn.neighbors import LocalOutlierFactor
import ClusteringAlgorithmsImportFile
from ClusteringAlgorithmInterface import ClusteringAlgorithm
from KMeans import KMeansAlgorithm
import GlobalParameters
import pandas as pd
from Dataset2 import Dataset2
from StatisticalTestC import sort_df_by_stat_test
import GlobalFunctions
# cluster with kmeans before and after removing anomalies.
anomaly_detection_algorithms_obj_list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list[0:1]
# anomaly_detection_algorithms_obj_list.append()
cluster_algo_obj_list = ClusteringAlgorithmsImportFile.clustering_algorithm_obj_list[0:3]

random_state_list = GlobalParameters.random_state_list

def get_csv_file_path(num_random_states, dataset_index, n_clusters, file_name = None):
    return GlobalFunctions.get_plot_file_path(file_name, dataset_index, "AnomalyDetecion")

def get_clean_df_local_outlier_factor(dataset):

    clf = LocalOutlierFactor()
    df = dataset.get_data_frame()
    results = clf.fit_predict(df)

    results = (results + 1) / 2
    results = results.astype(bool)

    return df.iloc[results]

# def get_clean_df_DBSCAN(dataset):

def n (dataset, random_state):
    n_clusters = dataset.get_n_clusters()
    algo_name_sil_score_dict = {}
    main_cluster_algo_obj = KMeansAlgorithm(nClusters = n_clusters, randomState = random_state, dataFrame = dataset.get_data_frame())
    silhouette_score = main_cluster_algo_obj.getSilhouetteScore()

    algo_name_sil_score_dict[f"all_data_{main_cluster_algo_obj.get_name()}"] = silhouette_score
    
    for cluster_algo_obj in cluster_algo_obj_list:
        cluster_algo_obj.setRandomState(GlobalParameters.random_state)
        cluster_algo_obj.setNClusters(n_clusters)
        _, clean_data_df = cluster_algo_obj.get_anomalous_dataframe_negative_silhouette_coefficients(dataset)
        
        
        main_cluster_algo_obj.setDataFrame(clean_data_df)
        main_cluster_algo_obj.createLabels()
        silhouette_score = main_cluster_algo_obj.getSilhouetteScore()
        algo_name_sil_score_dict[f"{cluster_algo_obj.get_name()}_anomaly_detection"] = silhouette_score
    
    clean_data_df = get_clean_df_local_outlier_factor(dataset)
    main_cluster_algo_obj.setDataFrame(clean_data_df)
    main_cluster_algo_obj.createLabels()
    silhouette_score = main_cluster_algo_obj.getSilhouetteScore()
    algo_name_sil_score_dict["KNN_anomaly_detection"] = silhouette_score

    return algo_name_sil_score_dict
def run_random_states(dataset):
    result_dict = {}
    num_random_stats = len(random_state_list)
    dataset_index = dataset.get_index()
    n_clusters = dataset.get_n_clusters()
    for random_state in random_state_list:
        dic = n(dataset, random_state)
        result_dict[random_state] = dic

    result_df = pd.DataFrame(result_dict).T

    stat_test_results_df, sorted_df = sort_df_by_stat_test(result_df)
    
    sorted_df.loc['mean'] = sorted_df.mean()
    # mean_df.plot.bar()
    ############################# save stat test results
    file_name = f"{num_random_stats}RandomStatesWith{n_clusters}ClustersStatisiticalTestResults.csv"
    csv_file_path = get_csv_file_path(num_random_stats, dataset_index, n_clusters, file_name)
    stat_test_results_df.to_csv(csv_file_path)
    
    ############################ save sorted.
    file_name = f"{num_random_stats}RandomStatesWith{n_clusters}ClustersSorted.csv"
    csv_file_path = get_csv_file_path(num_random_stats, dataset_index, n_clusters, file_name)
    sorted_df.to_csv(csv_file_path)

if __name__ == "__main__":
    dataset = Dataset2()
    run_random_states(dataset)