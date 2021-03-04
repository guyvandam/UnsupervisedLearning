from DataSet import DataSet
import GlobalParameters


class Dataset2(DataSet):
    
    def __init__(self):
        super().__init__(
            path = GlobalParameters.DATASET2_CSV_FILE_PATH,
            csv_seperator = ',',
            index = 2,
            n_classes = GlobalParameters.DATASET2_NUMBER_OF_CLASSES
        )

    def prepareDataset(self):
        super()._loadCSV(na_values = '?')
        
        ################################ extract ground truth column.
        self.ground_truth = self.df[self.df.columns[-1]]
        del self.df[self.df.columns[-1]]

        ################################ managable size so we don't need to downsample.

        ################################ drop rows with nan values
        # self.df.dropna(axis = 1, how = 'any', inplace = True)
        
        

# ds = Dataset2()
# ds.prepareDataset()
# print(df.get_data_frame)