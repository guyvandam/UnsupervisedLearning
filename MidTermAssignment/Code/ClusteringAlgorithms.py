from KMeans import KMeansAlgorithm
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm

global clusteringAlgorithmList

clusteringAlgorithmList = [
    KMeansAlgorithm(),
    GMMAlgorithm(),
    FuzzyCMeansAlgorithm(),
    AgglomerativeClusteringAlgorithm(),
    SpectralClusteringAlgorithm()
]
