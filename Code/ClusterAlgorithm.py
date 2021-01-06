from scipy.stats import entropy
from collections import Counter
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kneed import KneeLocator
import GlobalParameters
import numpy as np
import os


class ClusterAlgorithm():
    def __init__(self, nClusters: int, randomState=None, dataFrame=None):
        self.dataFrame = dataFrame
        self.nClusters = nClusters
        self.labels = None
        self.algorithmObject = None
        self.randomState = randomState
        self.name = None

    def setDataFrame(self, dataFrame):
        self.dataFrame = dataFrame

    def createLabels(self):
        if self.dataFrame is None:
            raise RuntimeError("Data not initialzed")
        self.labels = self.algorithmObject.fit_predict(self.dataFrame)

    def getLabels(self):
        return self.labels

    def getEntropy(self):
        if self.labels is None:
            self.createLabels()
        'need to give entropy the probability for each cluster'
        probabilityList = np.array(list(Counter(self.labels).values()))
        probabilityList = probabilityList / len(self.labels)
        return entropy(probabilityList)

    def getMutualInformation(self, groundTruth):
        return metrics.adjusted_mutual_info_score(self.labels, groundTruth)

    def getBestNumClusters(self, randomState: int, numClustersRange, datasetIndex) -> int:
        pass

    def getMinimaze(self):
        pass

    def getMinimazeLabel(self):
        pass

    # def noName(self, randomState, numClustersRange, datasetIndex, func, ylabelFunc,titleStartFunc,findKnee=True):
    #     imageList = []
    #     numClustersRange = range(2, numClustersRange+1)
    #     for nClusters in numClustersRange:
    #         print(f"{self.name} Clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
    #         self.__init__(nClusters, randomState, self.dataFrame)
    #         self.createLabels()
    #         imageList.append(func())

    #     if findKnee:
    #         knee = None
    #         try:
    #             kn = KneeLocator(numClustersRange, imageList, curve='convex', direction='decreasing')
    #             knee = kn.knee
    #         except Exception:
    #             print("Couldn't locate knee, returning None")

    #     plt.figure() # start a new figure.
    #     plt.xlabel(GlobalParameters.xlabel)
    #     plt.ylabel(ylabelFunc())
    #     plt.title(f"{titleStartFunc} for Dataset {datasetIndex} \n Using {self.name} Clustering With Random State {randomState}")
    #     plt.plot(numClustersRange, imageList, 'bx-')
    #     if findKnee and not knee is None: plt.vlines(kn.knee, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed')

    #     figPath = GlobalParameters.plotsLocation + f"{datasetIndex}\\{self.name}{nClusters}ClustersRandomState{randomState}.png"
    #     plt.savefig(figPath)
    #     return knee

    def getBestNumClustersElbowMethod(self, randomState: int, numClustersRange, datasetIndex):
        minimazeList = []
        numClustersRange = range(2, numClustersRange+1)
        for nClusters in numClustersRange:
            print(
                f"{self.name} Clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
            self.__init__(nClusters, randomState, self.dataFrame)
            self.createLabels()
            minimazeList.append(self.getMinimaze())

        knee = None
        try:
            kn = KneeLocator(numClustersRange, minimazeList,
                             curve='convex', direction='decreasing')
            knee = kn.knee
        except Exception:
            print("Couldn't locate knee, returning None")

        plt.figure()  # start a new figure.
        plt.xlabel(GlobalParameters.xlabel)
        plt.ylabel(self.getMinimazeLabel())
        plt.title(
            f"Elbow Method Showing Optimal Number Of Clusters for Dataset {datasetIndex} \n Using {self.name} Clustering With Random State {randomState}")
        plt.plot(numClustersRange, minimazeList, 'bx-')
        if not knee is None:
            plt.vlines(kn.knee, ymin=plt.ylim()[
                       0], ymax=plt.ylim()[1], linestyles='dashed')

        # figPath = GlobalParameters.plotsLocation + f"{datasetIndex}\\{self.name}{nClusters}ClustersRandomState{randomState}.png"
        # plt.savefig(figPath)

        Dir = GlobalParameters.plotsLocation + \
            f"{datasetIndex}\\{self.name}\\ElbowMethod\\{nClusters}Clusters"
        try:
            os.makedirs(Dir)
        except FileExistsError:
            pass
        plt.savefig(Dir + f"\\RandomState{randomState}.png")

        return knee

    def getBestNumClustersSilhouette(self, randomState, numClustersRange, datasetIndex):
        silhouetteList = []
        numClustersRange = range(2, numClustersRange+1)
        for nClusters in numClustersRange:
            print(
                f"{self.name} Clustering dataset {datasetIndex} with {nClusters} Clusters and Random state {randomState}")
            self.__init__(nClusters, randomState, self.dataFrame)
            self.createLabels()
            silhouetteList.append(silhouette_score(
                self.dataFrame, self.labels))

        plt.figure()  # start a new figure.
        plt.xlabel(GlobalParameters.xlabel)
        plt.ylabel("Silhouette Score")
        plt.title(
            f"Silhouette Score for Dataset {datasetIndex} \n Using {self.name} Clustering With Random State {randomState}")
        plt.plot(numClustersRange, silhouetteList, 'bx-')

        # parentDir = GlobalParameters.plotsLocation + f"{loadData.getDatasetIndex()}\\"
        Dir = GlobalParameters.plotsLocation + \
            f"{datasetIndex}\\{self.name}\\SilhouetteScore\\{nClusters}Clusters"
        try:
            os.makedirs(Dir)
        except FileExistsError:
            pass
        # figPath = GlobalParameters.plotsLocation + f"{loadData.getDatasetIndex()}\\MutualInfo{gt}{self.name}{nClusters}ClustersRandomState{randomState}.png"
        plt.savefig(Dir + f"\\RandomState{randomState}.png")
        # result[gt] = numClustersRange[np.argmax(mutualInfoList)]

        # figPath = GlobalParameters.plotsLocation + f"{datasetIndex}\\{self.name}{nClusters}ClustersRandomState{randomState}.png"
        # plt.savefig(figPath)
        return numClustersRange[np.argmax(silhouetteList)]

    def checkAgainstExternalClass(self, randomStates, externalClass):
        mutualInfoDict = {}
        nClusters = len(set(externalClass))
        for randomState in randomStates:
            self.__init__(nClusters, randomState, self.dataFrame)
            self.createLabels()
            mutualInfoDict[randomState] = self.getMutualInformation(externalClass)

        if len(set(mutualInfoDict.values())) == 1:print(f"Random State Doesn't make a change for {self.name}")
        return mutualInfoDict

    def getSilhouetteScoreList(self, randomStates):
        self.silhouetteList = []
        for randomState in randomStates:
            self.__init__(self.nClusters, randomState, self.dataFrame)
            self.createLabels()
            self.silhouetteList.append(
                silhouette_score(self.dataFrame, self.labels))

        if len(set(self.silhouetteList)) == 1:
            print(f"Random State Doesn't make a change for {self.name}")
        return self.silhouetteList

    # "Check Against the ground truth"
    # def getBestNumClustersExternalClass(self, randomState, numClustersRange, loadData):

    #     labelScoreDict = {gt:[] for gt in loadData.groundTruthColumns} # key - class, value - list
    #     numClustersRange = range(2, numClustersRange+1)
    #     result = {}
    #     for nClusters in numClustersRange:
    #         print(f"{self.name} Clustering dataset {loadData.getDatasetIndex()} with {nClusters} Clusters and Random state {randomState}")
    #         self.__init__(nClusters, randomState, self.dataFrame)
    #         self.createLabels()
    #         for label, content in loadData.getGroundTruth().iteritems():
    #             labelScoreDict[label].append(self.getMutualInformation(content))

    #     for gt, mutualInfoList in labelScoreDict.items():

    #         plt.figure() # start a new figure.
    #         plt.xlabel(GlobalParameters.xlabel)
    #         plt.ylabel("Mutual Information Between Labels And Extrenal Labels")
    #         plt.title(f"Mutual Information Between Prediction Labels And {gt} Labels for \nDataset {loadData.getDatasetIndex()} Using {self.name} Clustering With Random State {randomState}")
    #         plt.plot(numClustersRange, mutualInfoList, 'bx-')

    #         # parentDir = GlobalParameters.plotsLocation + f"{loadData.getDatasetIndex()}\\"
    #         Dir = GlobalParameters.plotsLocation + f"{loadData.getDatasetIndex()}\\{self.name}\\ExternalClass{gt}\\{nClusters}Clusters"
    #         try:
    #             os.makedirs(Dir)
    #         except FileExistsError:
    #             pass
    #         # figPath = GlobalParameters.plotsLocation + f"{loadData.getDatasetIndex()}\\MutualInfo{gt}{self.name}{nClusters}ClustersRandomState{randomState}.png"
    #         plt.savefig(Dir +f"\\RandomState{randomState}.png")
    #         result[gt] = numClustersRange[np.argmax(mutualInfoList)]
    #     return result
