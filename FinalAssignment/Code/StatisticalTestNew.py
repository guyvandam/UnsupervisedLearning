import ClusteringAlgorithmsImportFile
import DatasetsImportFile
import GlobalParameters
import GlobalFunctions

from StatisticalTestC import sort_df_by_stat_test
def get_csv_file_name(num_random_stats, n_clusters):
    file_name = f"{num_random_stats}RandomStatesSilhouetteScoresWith{n_clusters}Clusters.csv"
    return file_name

def get_csv_file_path(num_random_states, dataset_index, n_clusters, file_name = None):
    file_name = get_csv_file_name(num_random_states, n_clusters) if file_name == None else file_name
    return GlobalFunctions.get_plot_file_path(file_name, dataset_index, GlobalParameters.STATISTICAL_TEST_FOLDER_NAME)

class StatisticalTest():
    def __init__(self, randomStateList: list = GlobalParameters.random_state_list,
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


    

  
    def createCSV(self, dataset):
        """
        We perfrom the statistical test seeing in the paper. essentially finding the maximum with our newley defined order.
        We save the result in a dictionary which later transforms into a pandas.DataFrame which is saved to a CSV file.

        Args:
            dataSet (DataSet object): data-set we want to check the fitment between external labels and prediction labels.
        """
        num_random_stats = len(self.randomStateList)
        dataset_index = dataset.get_index()
        n_clusters = dataset.get_n_clusters()
        sill_scores_csv_file_path = get_csv_file_path(num_random_stats, dataset_index, n_clusters)

        result_df = GlobalFunctions.get_df_by_path(sill_scores_csv_file_path)

        stat_test_results_df, sorted_df = sort_df_by_stat_test(result_df)

        ############################# save stat test results
        file_name = f"{num_random_stats}RandomStatesWith{n_clusters}ClustersStatisiticalTestResults.csv"
        csv_file_path = get_csv_file_path(len(self.randomStateList), dataset_index, n_clusters, file_name)
        stat_test_results_df.to_csv(csv_file_path)
       
        ############################ save sorted.
        file_name = f"{num_random_stats}RandomStatesWith{n_clusters}ClustersSorted.csv"
        csv_file_path = get_csv_file_path(len(self.randomStateList), dataset_index, n_clusters, file_name)
        sorted_df.loc['mean'] = sorted_df.mean()
        sorted_df.to_csv(csv_file_path)


if __name__ == "__main__":
    ST = StatisticalTest()
    for ds in DatasetsImportFile.dataset_obj_list:
        ds.prepareDataset()
        ST.createCSV(ds)
