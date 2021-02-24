import time
import os
import matplotlib.pyplot as plt
import GlobalParameters
import ClusteringAlgorithms
from DataSet1 import DataSet1
from DataSet2 import DataSet2
from DataSet3 import DataSet3
import DataSets


class PlotClusters():

    def __init__(self, clusteringAlgorithmList: list = ClusteringAlgorithms.clusteringAlgorithmList):
        """

        Args:
            clusteringAlgorithmList (list, optional): Plot the data using these clustering algorithms. Defaults to ClusteringAlgorithms.clusteringAlgorithmList.
        """
        self.clusteringAlgorithms = clusteringAlgorithmList

    def plotAndSaveOne(self, dataSet):
        """
        Save a fig of the clustering result for each clustering algorithm in our list.

        Args:
            dataSet (DataSet object): the data set we want to plot the clusters of.
        """
        dataSet.prepareDataset()

        datasetIndex = dataSet.getDatasetIndex()
        data = dataSet.getDataFrame()
        algoNameSilhouetteScoreDict = {}
        randomState = GlobalParameters.randomState
        nrows = 2
        ncols = 3
        fontsize = 20
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(23, 10))
        fig.suptitle(
            f"Data-Set {datasetIndex} Clustering With Optimal \n Clusters Number And Random State {randomState}")
        # axes indexes
        i = 0
        j = 0

        for clusterAlgo in self.clusteringAlgorithms:
            clusterAlgo.setDataFrame(data)
            clusterAlgo.setNClustersDatasetIndex(datasetIndex)
            algoNameSilhouetteScoreDict[clusterAlgo.name] = clusterAlgo.getSilhouetteScoreList(
                [randomState])[0]  # function gets a list and returns a list.
            labels = clusterAlgo.getLabels()
            ax[i, j].scatter(data['dim1'], data['dim2'], c=labels)
            ax[i, j].set_title(
                f"{clusterAlgo.name} With {clusterAlgo.getNClusters()} Clusters", fontsize=fontsize)

            j += 1
            if j == ncols:
                j = 0
                i += 1

        # ---------- bar plot for silhouette score ----------
        ax[i, j].bar(list(algoNameSilhouetteScoreDict.keys()),
                     algoNameSilhouetteScoreDict.values())
        ax[i, j].set_title(f"Silhouette Score", fontsize=fontsize)

        # ---------- Save Plot ----------
        directory = os.path.join(
            os.getcwd(), f"Results\\Dataset{datasetIndex}\\ClusteringPlot")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        plt.savefig(
            directory + f"\\ClusteringPlotWithRandomState{randomState}.png")
        plt.close()


# Due to some memory interference issues with saving the figure and acessing the CSV file for loading the data, this function can't loop over the datasets and save their figures in one run.
plotClusters = PlotClusters()
# plotClusters.plotAndSaveOne(DataSet1())
# plotClusters.plotAndSaveOne(DataSet2())
plotClusters.plotAndSaveOne(DataSet3())
