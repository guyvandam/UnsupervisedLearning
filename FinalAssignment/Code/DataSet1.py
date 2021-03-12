from DatasetInterface import Dataset
GROUND_TRUTH_COLUMN = 'Class'

class Dataset1(Dataset):    
    def __init__(self):
        super().__init__(
            index = 1
        )

    def prepareDataset(self):
        ############################## load the csv file.
        super()._loadCSV(na_values = '?')

        ################################ drop the first row. full of zeros.
        self.df.drop(index = 0, axis = 0, inplace = True)

        ############################### drop user column.
        self.df.drop(columns = ['User', 'X11', 'Y11', 'Z11'], inplace = True)

        ################################ we subsample by removing all of the rows with less than 95% real values
        super().drop_rows_by_non_na_precent(non_na_precent = 95)
        
        ############################### fill na with row-wise median.
        self.df.fillna(value = self.df.median(axis = 0), axis = 0, inplace = True)
        
        ################################ extract ground truth column.
        self.ground_truth = self.df.pop(GROUND_TRUTH_COLUMN)
        
        super()._reduceDimensions()

if __name__ == "__main__":
    ds1 = Dataset1()
    ds1.prepareDataset()
    print("dataset1\n", ds1.get_data_frame())
    print(ds1.get_n_classes())


