from LoadDataSet1 import LoadDataSet1
from LoadDataSet3 import LoadDataSet3
from GMM import GMMAlgorithm
from FuzzyCMeans import FuzzyCMeansAlgorithm
from KMeans import KMeansAlgorithm
from AgglomerativeClustering import AgglomerativeClusteringAlgorithm
from SpectralClustering import SpectralClusteringAlgorithm
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd


class StatisticalTest():
    def __init__(self, loadData, clusterAlgorithms, randomStates):
        self.loadData = loadData
        self.clusterAlgorithms = clusterAlgorithms
        self.randomStates = randomStates
        self.result = {}
    
    def run(self):
        self.loadData.prepareDataset()

        for clusterAlgorithm in self.clusterAlgorithms:
            clusterAlgorithm.setDataFrame(self.loadData.getDataFrame())
        
        winner = self.clusterAlgorithms[0]
        winnerSilhouetteList = winner.getSilhouetteScoreList(self.randomStates)
        winnerAvg = np.mean(winnerSilhouetteList)
        
        for candidate in self.clusterAlgorithms[1:]:
            candidateSilhouetteList = candidate.getSilhouetteScoreList(self.randomStates)
            candidateAvg = np.mean(candidateSilhouetteList)
            
            stat, p = ttest_ind(winnerSilhouetteList, candidateSilhouetteList)
            columnKey = f"{winner.name}VS{candidate.name}"
            columnValue = {"pValue": p, "Stat": stat,
                        "Cluster1Avg": winnerAvg, "Cluster2Avg": candidateAvg}
            if p > 0.05:
                print(f"Probably the same mean between {winner.name} and {candidate.name}")
            else:
                print(f"Probably NOT the same mean between {winner.name} and {candidate.name}")
                if winnerAvg < candidateAvg:
                        print(f"Winner is now {candidate.name}")
                        winner = candidate
                        winnerSilhouetteList = candidateSilhouetteList
                        winnerAvg = candidateAvg

            self.result[columnKey] = columnValue

        self.result = pd.DataFrame(self.result)
        print(self.result)
        self.result.to_csv(f"StatisticalTestDataset{self.loadData.getDatasetIndex()}.csv")



ld = LoadDataSet1()

clusteringAlgorithms = [
    KMeansAlgorithm(nComponents=4),
    GMMAlgorithm(nComponents=4),
    FuzzyCMeansAlgorithm(nComponents=5),
    AgglomerativeClusteringAlgorithm(
        nComponents=3),
    SpectralClusteringAlgorithm(nComponents=3)
]
randomStates = [68, 94, 60, 17, 1, 63, 11, 77, 4, 45, 24, 15, 3, 42, 32, 21, 62, 7, 0, 78, 9, 61, 57, 84, 19, 56, 2, 14, 87, 46]
ST = StatisticalTest(ld,clusteringAlgorithms,randomStates)
ST.run()


# ld = LoadDataSet1()
#
# ld.prepareDataset()
#
# clusteringAlgorithms = [
#     KMeansAlgorithm(nComponents=4),
#     GMMAlgorithm(nComponents=4),
#     FuzzyCMeansAlgorithm(nComponents=5),
#     AgglomerativeClusteringAlgorithm(
#         nComponents=3),
#     SpectralClusteringAlgorithm(nComponents=3)
# ]
#
# randomStates = [0, 1]  # need to be something like 30..
# winner = clusteringAlgorithms[0]
# winnerSilhouette = winner.getSilhouetteScoreList(randomStates)
# winnerAvg = np.mean(winnerSilhouette)
# result = {}
# for i in range(1, len(clusteringAlgorithms)):
#     candidate = clusteringAlgorithms[i]
#
#     candidateSilhouette = candidate.getSilhouetteScoreList(randomStates)
#     candidateAvg = np.mean(candidateSilhouette)
#
#     # check who is bigger
#     stat, p = ttest_ind(winnerSilhouette, candidateSilhouette)
#     key = f"{winner.name}VS{candidate.name}"
#     columnDict = {"pValue": p, "stat": stat,
#                   "Cluster1Avg": winnerAvg, "Cluster2Avg": candidateAvg}
#     if p > 0.05:
#         print(f"Probably the same mean between {winner.name} and {candidate.name}")
#     else:
#         print(f"Probably not the same mean between {winner.name} and {candidate.name}")
#         if winnerAvg < candidateAvg:
#                 print(f"Winner is now {candidate.name}")
#                 winner = candidate
#                 winnerSilhouette = candidateSilhouette
#                 winnerAvg = candidateAvg
#     result[key] = columnDict
#
# result = pd.DataFrame(result)
# print(result)
# result.to_csv(f"StatisticalTestDataset{ld.getDatasetIndex()}.csv")


