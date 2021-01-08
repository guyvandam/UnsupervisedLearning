import os
import pandas as pd
from DataSet1 import DataSet1
from DataSet2 import DataSet2
from DataSet3 import DataSet3
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
            maxList = result.idxmax(axis=1)
            minList = result.idxmin(axis=1)
            result['Max'] = maxList
            result['Min'] = minList
            

            # ---------- Save results in a CSV file ----------
            directory = os.path.join(os.getcwd(), f"Results\\Dataset{self.loadData.getDatasetIndex()}\\ExternalLabels")
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass
            result.to_csv(directory+f"\\{label}ClassifierWith{nClusters}ClustersAnd{len(self.randomStates)}RandomStates.csv")



dataSetList = [DataSet1(), DataSet2(), DataSet3()]

# for ld in loadDataList[2:]:
for ld in dataSetList:
    ld.prepareDataset()
    print(f"Running On Dataset {ld.getDatasetIndex()}")
    clusteringAlgorithms = [
        KMeansAlgorithm(dataFrame=ld.getDataFrame()),
        GMMAlgorithm(dataFrame=ld.getDataFrame()),
        FuzzyCMeansAlgorithm(dataFrame=ld.getDataFrame()),
        AgglomerativeClusteringAlgorithm(dataFrame=ld.getDataFrame()),
        SpectralClusteringAlgorithm(dataFrame=ld.getDataFrame())
    ]

    # fec = FitExternalClass(ld, clusteringAlgorithms[0:2], GlobalParameters.randomStates[0:2])
    fec = FitExternalClass(ld, clusteringAlgorithms, GlobalParameters.randomStates)
    fec.createCSV()
