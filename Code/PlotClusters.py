import matplotlib.pyplot as plt
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from KMeans import KMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm
from LoadDataSet1 import LoadDataSet1
from LoadDataSet3 import LoadDataSet3
from LoadDataSet2 import LoadDataSet2
import GlobalParameters
ld = LoadDataSet3()

clusteringAlgorithms = [
        KMeansAlgorithm(),
        GMMAlgorithm(),
        FuzzyCMeansAlgorithm(),
        AgglomerativeClusteringAlgorithm(),
        SpectralClusteringAlgorithm()
    ]

    

ld.prepareDataset()

datasetIndex = ld.getDatasetIndex()
data = ld.getDataFrame()
algoNameSilhouetteScoreDict = {}
randomState = GlobalParameters.random_state
for clusterAlgo in clusteringAlgorithms:
    clusterAlgo.setDataFrame(data)
    clusterAlgo.setNumClustersDatasetIndex(datasetIndex)
    algoNameSilhouetteScoreDict[clusterAlgo.name] = clusterAlgo.getSilhouetteScoreList([randomState])[0] # function gets a list and returns a list.


plt.title(f"Silhouette Score With Optimal Cluster Number For \n Data-Set {datasetIndex} And Random State {randomState}")
plt.bar(list(algoNameSilhouetteScoreDict.keys()), algoNameSilhouetteScoreDict.values())

# ---------- Save Plot ----------
# plt.savefig()
plt.show()


# clusteringAlgorithm.createLabels()
# labels = clusteringAlgorithm.getLabels()

# plt.scatter(data['dim1'],data['dim2'],c=labels)

# plt.show()



