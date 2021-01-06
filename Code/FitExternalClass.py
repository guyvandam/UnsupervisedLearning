import os
import pandas as pd
from LoadDataSet1 import LoadDataSet1
from LoadDataSet2 import LoadDataSet2
from LoadDataSet3 import LoadDataSet3
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from KMeans import KMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm
import GlobalParameters

class FitExternalClass():

    def __init__(self, loadData, clusteringAlgorithms, randomStates):
        self.loadData = loadData
        self.clusteringAlgorithms = clusteringAlgorithms
        self.randomStates = randomStates

    def createCSV(self):

        for label, content in self.loadData.getGroundTruth().iteritems():
            nClusters = len(set(content))
            print(f"Checking {label} Classifier With {nClusters} Clusters")
            result = {}
            for clusteringAlgorithm in self.clusteringAlgorithms:
                result[clusteringAlgorithm.name] = clusteringAlgorithm.checkAgainstExternalClass(
                    self.randomStates, content)

            result = pd.DataFrame(result)
            print(result)

            # ---------- Save results in a CSV file ----------
            directory = os.path.join(os.getcwd(), f"Results\\Dataset{self.loadData.getDatasetIndex()}")
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass
            result.to_csv(directory+f"\\{label}ClassifierWith{nClusters}Clusters.csv")


# ld = LoadDataSet2()
loadDataList = [LoadDataSet1(),LoadDataSet2(),LoadDataSet3()]

for ld in loadDataList[2:]:
    ld.prepareDataset()
    print(ld.getDatasetIndex())
    clusteringAlgorithms = [
        KMeansAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        GMMAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        FuzzyCMeansAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        AgglomerativeClusteringAlgorithm(
            nComponents=1, dataFrame=ld.getDataFrame()),
        SpectralClusteringAlgorithm(nComponents=1, dataFrame=ld.getDataFrame())
    ]

    fec = FitExternalClass(ld, clusteringAlgorithms, GlobalParameters.randomStates[0:1])
    fec.createCSV()
