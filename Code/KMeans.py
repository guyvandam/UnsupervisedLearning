from sklearn.cluster import KMeans

from ClusterAlgorithm import ClusterAlgorithm


class KMeansAlgorithm(ClusterAlgorithm):
    def __init__(self, nClusters=None, randomState=None, dataFrame=None):
        super().__init__(nClusters, randomState, dataFrame=dataFrame)
        self.algorithmObject = KMeans(
            n_clusters=self.nClusters, random_state=self.randomState) # all the parameters we need and learned in class.
        self.name = "KMeans"


    def getMinimaze(self):
        return self.algorithmObject.inertia_


    def getMinimazeLabel(self):
        return "Inertia - Sum Of Squared Distances"
    
    def getBestNumClusters(self, randomState, numClustersRange, datasetIndex):
        # inertiaList = []
        # numClustersRange = range(2, numClustersRange+1)
        # for nClusters in numClustersRange:
        #     print(f"{self.name} Clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
        #     self.algorithmObject = KMeans(n_clusters=nClusters, random_state=randomState)
        #     self.createLabels()
        #     inertiaList.append(self.algorithmObject.inertia_)

        # self.kn = KneeLocator(numClustersRange, inertiaList,curve='convex', direction='decreasing')

        # plt.figure() # start a new figure.
        # plt.xlabel(GlobalParameters.xlabel)
        # plt.ylabel('Sum of squared distances (inertia)')
        # plt.title("The Elbow Method Showing optimal k for dataset " + str(datasetIndex))
        # plt.plot(numClustersRange, inertiaList, 'bx-')
        # plt.vlines(self.kn.knee, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed')
    
        # path = GlobalParameters.plotsLocation + str(datasetIndex) + f"\\{self.name}" + str(nClusters) + f"ClustersRandomState{randomState}.png"
        # plt.savefig(path)
        # return self.kn.knee
        return super().getBestNumClustersElbowMethod(randomState, numClustersRange, datasetIndex)
