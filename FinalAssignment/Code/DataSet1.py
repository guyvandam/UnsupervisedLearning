from DataSet import DataSet
import GlobalParameters


class Dataset1(DataSet):
    
    def __init__(self):
        super().__init__(
            path = GlobalParameters.DATASET1_CSV_FILE_PATH,
            csv_seperator = ',',
            index = 1,
            n_classes = GlobalParameters.DATASET1_NUMBER_OF_CLASSES
        )

    def prepareDataset(self):
        super()._loadCSV()

        print(self.df)

ds1 = Dataset1()
ds1.prepareDataset()