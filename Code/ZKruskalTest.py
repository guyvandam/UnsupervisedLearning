from LoadDataSet1 import LoadDataSet1
from LoadDataSet3 import LoadDataSet3
from PCA import PCAAlgorithm
from KMeans import KMeansAlgorithm
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from scipy.stats import kruskal
from collections import Counter
import numpy as np

def sortByOccurrences(lst:list):
    result = np.array([])
    occurrencesDict = Counter(lst)
    i = 0
    print(occurrencesDict)
    for key, value in occurrencesDict.items():
        result = np.append(result, i*[key])
        i+=1

    return result

nClusters = 47
# ld = LoadDataSet1(nrows=10)
ld = LoadDataSet3()
ld.prepareDataset()


kmeans = KMeansAlgorithm(nClusters)
gmm = GMMAlgorithm(nClusters)
fuzzy = FuzzyCMeans(nClusters)

clusteringAlgorithms = [kmeans, gmm, fuzzy]
labels = []
for ca in clusteringAlgorithms:
    ca.createLabels(ld.getDataFrame())
    labels.append(sortByOccurrences(ca.getLabels()))

labels.append(ld.getDataFrame()['country']) 
# kmeanslabels = sortByOccurrences(kmeans.getLabels())
# gmmlabels = sortByOccurrences(gmm.getLabels())


# stat, p = kruskal(kmeanslabels, gmmlabels)
stat, p = kruskal(*labels)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


        