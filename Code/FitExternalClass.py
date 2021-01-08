import os

import matplotlib.pyplot as plt
import pandas as pd

import ClusteringAlgorithms
import DataSets
import GlobalParameters
from DataSet1 import DataSet1
from DataSet2 import DataSet2
from DataSet3 import DataSet3

class FitExternalClass():

    def __init__(self, randomStateList=None, clusteringAlgorithms=None):
        self.clusteringAlgorithms =  ClusteringAlgorithms.clusteringAlgorithmsList if clusteringAlgorithms is None else clusteringAlgorithms
        self.randomStateList = GlobalParameters.randomStates if randomStateList is None else randomStateList


    

    def createCSV(self, dataSet):
        dataSet.prepareDataset()
        for label, content in dataSet.getGroundTruth().iteritems():
            nClusters = len(set(content))
            print(f"Checking {label} Classifier With {nClusters} Clusters")
            result = {}
            for clusteringAlgorithm in self.clusteringAlgorithms:
                clusteringAlgorithm.setDataFrame(dataSet.getDataFrame())
                result[clusteringAlgorithm.name] = clusteringAlgorithm.checkAgainstExternalClass(self.randomStateList, content)
            
            # ---------- Results into a DataFrame and add min and max columns ----------
            resultDF = pd.DataFrame(result)
            maxList = resultDF.idxmax(axis=1)
            minList = resultDF.idxmin(axis=1)
            resultDF['Max'] = maxList
            resultDF['Min'] = minList
            
            # ---------- Make Directory ----------
            directory = os.path.join(os.getcwd(), f"Results\\Dataset{dataSet.getDatasetIndex()}\\ExternalLabels")
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass


            # ---------- Save results in a CSV file ----------
            resultDF.to_csv(directory+f"\\{label}ClassifierWith{nClusters}ClustersAnd{len(self.randomStateList)}RandomStates.csv")

            # ---------- Save bar plot ----------
            algoNameAverageFitDict = {name:fitmentDict['Average'] for name, fitmentDict in result.items()}
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.bar(list(algoNameAverageFitDict.keys()),algoNameAverageFitDict.values())
            fig.suptitle(f"Average Of Kullback-Leibler Divergence Between Prediction Labels And External \n '{label}' Classifier With {nClusters} Clusters For Data-Set {dataSet.getDatasetIndex()} Across {len(self.randomStateList)} Random States")
            fig.savefig(directory + f"\\{label}ClassifierWith{nClusters}ClustersAnd{len(self.randomStateList)}RandomStates.png")
            plt.close()


"""
Due to some memory interference issues with saving the figure and acessing the CSV file for loading the data, this function can't loop over the datasets and save their figures
"""
# ds = DataSet1()
ds = DataSet2()
# ds = DataSet3()
fec = FitExternalClass()
fec.createCSV(ds)
