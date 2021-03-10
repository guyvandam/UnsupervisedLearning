from KMeans import KMeansAlgorithm
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm

global clustering_algorithm_obj_list

clustering_algorithm_obj_list = [
    KMeansAlgorithm(),
    GMMAlgorithm(),
    FuzzyCMeansAlgorithm(),
    AgglomerativeClusteringAlgorithm(),
    SpectralClusteringAlgorithm()
]
