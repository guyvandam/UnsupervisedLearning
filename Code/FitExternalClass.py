import os

import matplotlib.pyplot as plt
import pandas as pd

import ClusteringAlgorithms
import DataSets
import GlobalParameters


class FitExternalClass():

    def __init__(self, loadData, randomStates, clusteringAlgorithms=None):
        self.loadData = loadData
        self.clusteringAlgorithms =  ClusteringAlgorithms.clusteringAlgorithmsList if clusteringAlgorithms is None else clusteringAlgorithms
        self.randomStates = randomStates

    def createCSV(self):

        for label, content in self.loadData.getGroundTruth().iteritems():
            nClusters = len(set(content))
            print(f"Checking {label} Classifier With {nClusters} Clusters")
            result = {}
            for clusteringAlgorithm in self.clusteringAlgorithms:
                clusteringAlgorithm.setDataFrame(self.loadData.getDataFrame())
                result[clusteringAlgorithm.name] = clusteringAlgorithm.checkAgainstExternalClass(self.randomStates, content)
            
            # ---------- Results into a DataFrame and add min and max columns ----------
            resultDF = pd.DataFrame(result)
            maxList = resultDF.idxmax(axis=1)
            minList = resultDF.idxmin(axis=1)
            resultDF['Max'] = maxList
            resultDF['Min'] = minList
            
            # ---------- Make Directory ----------
            directory = os.path.join(os.getcwd(), f"Results\\Dataset{self.loadData.getDatasetIndex()}\\ExternalLabels")
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass


            # ---------- Save results in a CSV file ----------
            resultDF.to_csv(directory+f"\\{label}ClassifierWith{nClusters}ClustersAnd{len(self.randomStates)}RandomStates.csv")

            # ---------- Save bar plot ----------
            algoNameAverageFitDict = {name:fitmentDict['Average'] for name, fitmentDict in result.items()}
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.bar(list(algoNameAverageFitDict.keys()),algoNameAverageFitDict.values())
            fig.suptitle(f"Average Of Kullback-Leibler Divergence Between Prediction Labels And External \n '{label}' Classifier With {nClusters} Clusters For Data-Set {self.loadData.getDatasetIndex()} Across {len(self.randomStates)} Random States")
            fig.savefig(directory + f"\\{label}ClassifierWith{nClusters}ClustersAnd{len(self.randomStates)}RandomStates.png")
            plt.close()



for ld in DataSets.dataSetList[0:1]:

    ld.prepareDataset()
    print(f"Running On Dataset {ld.getDatasetIndex()}")
    
    fec = FitExternalClass(ld, GlobalParameters.randomStates[0:2])
    fec.createCSV()
