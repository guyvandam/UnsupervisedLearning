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
        super()._loadCSV()

        print(self.df)

ds = Dataset2()
ds.prepareDataset()