from DatasetInterface import Dataset

from SpectralClustering import SpectralClusteringAlgorithm

class Dataset2(Dataset):
    
    def __init__(self):
        super().__init__(
            index = 2
        )

    def prepareDataset(self):
        super()._loadCSV(na_values = '?')
        
        ################################ extract ground truth column.
        self.ground_truth = self.df[self.df.columns[-1]]
        del self.df[self.df.columns[-1]]

        ################################ managable size so we don't need to downsample.

        ################################ drop rows with nan values
        # self.df.dropna(axis = 1, how = 'any', inplace = True)
        super()._reduceDimensions()
        
if __name__ == "__main__":
    ds = Dataset2()
    ds.prepareDataset()
    # print(ds.get_data_frame())
    # print(ds.get_n_classes())

    # spec = SpectralClusteringAlgorithm(nClusters = 2, dataFrame = ds.get_data_frame())
    # spec.createLabels()
    # print(spec.getLabels())