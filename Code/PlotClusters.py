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

    def __init__(self, clusteringAlgorithms):
        self.clusteringAlgorithms = clusteringAlgorithms

    """
    Due to some memory interference issues with saving the figure and acessing the CSV file for loading the data, this function can't loop over the datasets and save their figures
    """

    def plotAndSaveOne(self, ld):
        
        ld.prepareDataset()

        datasetIndex = ld.getDatasetIndex()
        data = ld.getDataFrame()
        algoNameSilhouetteScoreDict = {}
        randomState = GlobalParameters.random_state
        nrows = 3
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
        fig.suptitle(f"Data-Set {datasetIndex} Clustering With Optimal \n Clusters Number And Random State {randomState}")
        # axes indexes
        i = 0
        j = 0

        for clusterAlgo in self.clusteringAlgorithms:
            clusterAlgo.setDataFrame(data)
            clusterAlgo.setNumClustersDatasetIndex(datasetIndex)
            algoNameSilhouetteScoreDict[clusterAlgo.name] = clusterAlgo.getSilhouetteScoreList(
                [randomState])[0]  # function gets a list and returns a list.
            labels = clusterAlgo.getLabels()
            ax[i, j].scatter(data['dim1'], data['dim2'], c=labels)
            ax[i, j].set_title(
                f"{clusterAlgo.name} With {clusterAlgo.getNClusters()} Clusters")

            j += 1
            if j == ncols:
                j = 0
                i += 1

        # ---------- bar plot for silhouette score ----------
        ax[i, j].bar(list(algoNameSilhouetteScoreDict.keys()),
                    algoNameSilhouetteScoreDict.values())
        ax[i, j].set_title(f"Silhouette Score")

        # ---------- Save Plot ----------
        directory = os.path.join(os.getcwd(), f"Results\\Dataset{datasetIndex}\\ClusteringPlot")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        plt.savefig(directory + f"\\ClusteringPlotWithRandomState{randomState}.png")
        # plt.show()
        plt.close()


plotClusters = PlotClusters(ClusteringAlgorithms.clusteringAlgorithmsList)
plotClusters.plotAndSaveOne(DataSet1())
# plotClusters.plotAndSaveOne(DataSet2())
# plotClusters.plotAndSaveOne(DataSet3())
