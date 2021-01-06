import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import entropy
from sklearn import metrics
from sklearn.metrics import silhouette_score

import GlobalParameters


class ClusterAlgorithm():
    def __init__(self, nClusters: int, randomState=None, dataFrame=None):
        self.dataFrame = dataFrame
        self.nClusters = nClusters
        self.labels = None
        self.algorithmObject = None
        self.randomState = randomState
        self.name = None
        self.optimalClustersNumberDict = {} # key - dataset index, value, number of clusters

    def setDataFrame(self, dataFrame):
        self.dataFrame = dataFrame

    def setOptimalClustersNumberDict(self, optDict=None):
        path = str(os.path.join(os.getcwd(), "Results\\OptimalClustersNumber.csv"))
        numOfClustersDF = pd.read_csv(path)
        nClustersList = list(numOfClustersDF[self.name])
        self.optimalClustersNumberDict = dict(zip(range(1,len(nClustersList)+1), nClustersList))

    def createLabels(self):
        if self.dataFrame is None:
            raise RuntimeError("Data not initialzed")
        self.labels = self.algorithmObject.fit_predict(self.dataFrame)

    def getLabels(self) -> np.ndarray:
        print("lables type is " + type(self.labels))
        return self.labels

    def getEntropy(self):
        if self.labels is None:
            self.createLabels()
        'need to give entropy the probability for each cluster'
        probabilityList = np.array(list(Counter(self.labels).values()))
        probabilityList = probabilityList / len(self.labels)
        return entropy(probabilityList)

    def getMutualInformation(self, groundTruth) -> float:
        return metrics.adjusted_mutual_info_score(self.labels, groundTruth)

    def getBestNumClusters(self, randomState: int, numClustersRange, datasetIndex) -> int:
        pass

    def getMinimaze(self):
        pass

    def getMinimazeLabel(self):
        pass
    
    def setNumClustersDatasetIndex(self, datasetIndex):
        self.__init__(self.optimalClustersNumberDict[datasetIndex], self.randomState, self.dataFrame)

    def setNumClusters(self, nClusters):
        self.__init__(nClusters, self.randomState, self.dataFrame)

    def getBestNumClustersElbowMethod(self, randomState: int, numClustersRange, datasetIndex):
        minimazeList = []
        numClustersRange = range(2, numClustersRange+1)
        for nClusters in numClustersRange:
            print(f"{self.name} Clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
            self.__init__(nClusters, randomState, self.dataFrame)
            self.createLabels()
            minimazeList.append(self.getMinimaze())

        knee = None
        try:
            kn = KneeLocator(numClustersRange, minimazeList,curve='convex', direction='decreasing')
            knee = kn.knee
        except Exception:
            print("Couldn't locate knee, returning None")

        plt.figure()  # start a new figure.
        plt.xlabel(GlobalParameters.xlabel)
        plt.ylabel(self.getMinimazeLabel())
        plt.title(f"Elbow Method Showing Optimal Number Of Clusters For Data-Set {datasetIndex} \n Using {self.name} Clustering With Random State {randomState}")
        plt.plot(numClustersRange, minimazeList, 'bx-')
        
        if not knee is None:
            plt.vlines(kn.knee, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed')

        directory = os.path.join(os.getcwd(), f"Results\\Dataset{datasetIndex}\\OptimalClustersNumber\\Plots\\{self.name}\\{nClusters}ClustersRange")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        plt.savefig(directory + f"\\RandomState{randomState}.png")

        return knee

    def getBestNumClustersSilhouette(self, randomState, numClustersRange, datasetIndex):
        silhouetteList = []
        numClustersRange = range(2, numClustersRange+1)
        for nClusters in numClustersRange:
            print(f"{self.name} Clustering dataset {datasetIndex} with {nClusters} Clusters and Random state {randomState}")
            self.__init__(nClusters, randomState, self.dataFrame)
            self.createLabels()
            silhouetteList.append(silhouette_score(
                self.dataFrame, self.labels))

        plt.figure()  # start a new figure.
        plt.xlabel(GlobalParameters.xlabel)
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette Score For Data-Set {datasetIndex} \n Using {self.name} Clustering With Random State {randomState}")
        plt.plot(numClustersRange, silhouetteList, 'bx-')

        directory = os.path.join(os.getcwd(), f"Results\\Dataset{datasetIndex}\\OptimalClustersNumber\\Plots\\{self.name}\\{nClusters}ClustersRange")
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        plt.savefig(directory + f"\\RandomState{randomState}.png")

        return numClustersRange[np.argmax(silhouetteList)]

    def checkAgainstExternalClass(self, randomStateList, externalClass):
        mutualInfoDict = {} # key - randoom state, value mutual info score.
        nClusters = len(set(externalClass))
        for randomState in randomStateList:
            self.__init__(nClusters, randomState, self.dataFrame)
            self.createLabels()
            mutualInfoDict[str(randomState)] = self.getMutualInformation(externalClass)

        mutualInfoDict['Average'] = np.mean(list(mutualInfoDict.values()))
        if len(set(mutualInfoDict.values())) == 1: print(f"Random State Doesn't make a change for {self.name}")
        return mutualInfoDict

    def getSilhouetteScoreList(self, randomStateList):
        self.silhouetteList = []
        for randomState in randomStateList:
            self.__init__(self.nClusters, randomState, self.dataFrame)
            self.createLabels()
            self.silhouetteList.append(
                silhouette_score(self.dataFrame, self.labels))

        if len(set(self.silhouetteList)) == 1: print(f"Random State Doesn't make a change for {self.name}")
        return self.silhouetteList
