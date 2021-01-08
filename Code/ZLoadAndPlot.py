from DataSet import DataSet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from DimReductionAlgorithm import DimReductionAlgorithm
from ClusterAlgorithm import ClusterAlgorithm
from sklearn.metrics import silhouette_samples, silhouette_score

class DoAll():
    def __init__(self, loadData:DataSet, clusterAlgorithm, dimReductionAlgorithm):
        self.dataFrame = None
        self.loadData = loadData
        self.clusterAlgorithm = clusterAlgorithm
        self.dimReductionAlgorithm = dimReductionAlgorithm
    
    def initialize(self):
        self.dataFrame = self.loadData.getDataFrame()
    
    def reduceDimensions(self):
        self.dimReductionAlgorithm.reduceDim(self.dataFrame)
        self.dataFrame = self.dimReductionAlgorithm.getDataFrame()

    def createLabels(self):
        self.clusterAlgorithm.createLabels(self.dataFrame)
    
    def addLabels(self):
        self.dataFrame['cluster'] = self.clusterAlgorithm.getLabels()

    def getLabels(self):
        return self.clusterAlgorithm.getLabels()

    def run(self):
        self.initialize()
        self.createLabels()
        self.reduceDimensions()
        self.addLabels()
        

    def plotData(self):
        self.run()
        color=['blue','green','cyan', 'black']
        for k in range(0,4):
            printdata = self.dataFrame[self.dataFrame['cluster']==k]
            plt.scatter(printdata['dim1'],printdata['dim2'],c=color[k])

        plt.show()
      