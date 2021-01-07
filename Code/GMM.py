from sklearn.mixture import GaussianMixture

from ClusterAlgorithm import ClusterAlgorithm


class GMMAlgorithm(ClusterAlgorithm):

    def __init__(self, nClusters=None, randomState=None, dataFrame=None):
        super().__init__(nClusters, randomState, dataFrame=dataFrame)
        self.algorithmObject = GaussianMixture(
            n_components=self.nClusters, random_state=self.randomState) # all the parameters we need and learned in class. Other parameters are what we disire by default.
        self.name = "GMM"

    def getMinimaze(self):
        return self.algorithmObject.bic(self.dataFrame)

    def getMinimazeLabel(self):
        return "BIC - Bayesian Information Criterion"

    def getBestNumClusters(self, randomState, numClustersRange, datasetIndex):
        # bicList = []
        # numClustersRange = range(2,numClustersRange+1)
        # for nClusters in numClustersRange:
        #     print(f"{self.name} Clustering dataset {datasetIndex} with {nClusters} clusters and Random state {randomState}")
        #     self.algorithmObject = GaussianMixture(n_components=nClusters, random_state=randomState)
        #     self.createLabels()
        #     bicList.append(self.algorithmObject.bic(self.dataFrame))

        # kn = KneeLocator(numClustersRange, bicList,curve='convex', direction='decreasing')

        # plt.figure() # start a new figure.
        # plt.xlabel(GlobalParameters.xlabel)
        # plt.ylabel("BIC - Bayesian Information criterion")
        # plt.title("Elbow Method Showing optimal k for dataset " + str(datasetIndex))
        # plt.plot(numClustersRange, bicList, 'bx-')
        # plt.vlines(kn.knee, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed')

        # path = GlobalParameters.plotsLocation + str(datasetIndex) + f"\\{self.name}" + str(nClusters) + f"ClustersRandomState{randomState}.png"
        # plt.savefig(path)
        # plt.close()
        # return kn.knee

        return super().getBestNumClustersElbowMethod(randomState, numClustersRange, datasetIndex)
