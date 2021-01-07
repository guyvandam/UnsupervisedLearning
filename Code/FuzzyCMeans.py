import numpy as np
import skfuzzy as fuzz

from ClusterAlgorithm import ClusterAlgorithm


class FuzzyCMeansAlgorithm(ClusterAlgorithm):
    def __init__(self, nClusters=None, randomState=None, dataFrame=None):
        super().__init__(nClusters, randomState, dataFrame=dataFrame)
        self.fpc = None
        self.name = "FuzzyCMeans"
         
    def createLabels(self):
        # u is the probability for each point to be in the cluster
        _ , u, _ , _ , _ , _ , fpc = fuzz.cluster.cmeans(np.array(self.dataFrame).T, self.nClusters, m=2, error=0.005,maxiter=1000,seed=self.randomState)
        self.labels = np.argmax(u, axis=0)
        self.fpc = fpc
        
    def getMinimaze(self):
        return self.fpc
    
    def getMinimazeLabel(self):
        return "FPC - Fuzzy Partition Coefficient"

    def getBestNumClusters(self, randomState, numClustersRange, datasetIndex):
        # fpcList = []
        # numClustersRange = range(2,numClustersRange+1)
        # for nClusters in numClustersRange:
        #     print(f"fuzzyCMeans clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
        #     self.randomState = randomState
        #     self.nClusters = nClusters
        #     self.createLabels()
        #     fpcList.append(self.fpc)
        
        # kn = KneeLocator(numClustersRange, fpcList, curve='convex', direction='decreasing')
        
        # plt.figure() # start a new figures.
        # plt.xlabel(GlobalParameters.xlabel)
        # plt.ylabel("FPC - Fuzzy Partition Coefficient")
        # plt.title("The Elbow Method Showing optimal k for dataset " + str(datasetIndex))
        # plt.plot(numClustersRange, fpcList, 'bx-')
        # plt.vlines(kn.knee, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed')

        # path = GlobalParameters.plotsLocation + str(datasetIndex) + f"\\{self.name}" + str(nClusters) + f"ClustersRandomState{randomState}.png"
        # plt.savefig(path)

        # return kn.knee 
        return super().getBestNumClustersElbowMethod(randomState,numClustersRange, datasetIndex)
