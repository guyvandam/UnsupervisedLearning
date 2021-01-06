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
            result['Best'] = result.idxmax(axis=1)

            # ---------- Save results in a CSV file ----------
            directory = os.path.join(os.getcwd(), f"Results\\Dataset{self.loadData.getDatasetIndex()}\\ExternalLabels")
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass
            result.to_csv(directory+f"\\{label}ClassifierWith{nClusters}ClustersAnd{len(self.randomStates)}RandomStates.csv")



loadDataList = [LoadDataSet1(),LoadDataSet2(),LoadDataSet3()]

for ld in loadDataList[2:]:
    ld.prepareDataset()
    print(f"Running On Dataset {ld.getDatasetIndex()}")
    clusteringAlgorithms = [
        KMeansAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        GMMAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        FuzzyCMeansAlgorithm(nComponents=1, dataFrame=ld.getDataFrame()),
        AgglomerativeClusteringAlgorithm(
            nComponents=1, dataFrame=ld.getDataFrame()),
        SpectralClusteringAlgorithm(nComponents=1, dataFrame=ld.getDataFrame())
    ]

    fec = FitExternalClass(ld, clusteringAlgorithms[0:2], GlobalParameters.randomStates[0:2])
    fec.createCSV()
