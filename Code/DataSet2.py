from LoadData import LoadData
import pandas as pd
import numpy as np
import GlobalParameters
set2Path = GlobalParameters.set2Path


class LoadDataSet2(LoadData):

    def __init__(self, nrows=None):
        super().__init__(path=set2Path, seperator=",", datasetIndex=2, nrows=nrows)
        self.groundTruthColumns = ['race', 'gender']

    def prepareDataset(self):
        # doesn't tell us anything about the data.
        columnsToDelete = ['encounter_id', 'patient_nbr', 'payer_code']
        allNumberColumns = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
                            'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
        self.dataFrame = pd.read_csv(self.path, sep=self.seperator, nrows=self.nrows, na_values="?")

        # We'll subsample the data due to runtime and memory issues. We want about 10,000 points/rows.
        numColumns = len(self.dataFrame.columns)
        # "Thresh - Require that many non-NA values". We keep a row with at least 50-1 non-NA values. this gives us about 28000 points.
        self.dataFrame.dropna(thresh=numColumns-1, inplace=True)
        # taking 14000 point/rows. With a fixed random state.
        self.dataFrame = self.dataFrame.sample(
            n=14000, random_state=GlobalParameters.random_state)

        for columnIndex in self.dataFrame.columns:
            if not columnIndex in allNumberColumns:
                column = self.dataFrame[columnIndex]
                unique = set(column)
                if np.NaN in unique:
                    unique.remove(np.NaN)

                # only one value in a column. Doesn't tell us anything about that data.
                if len(unique) == 1 or columnIndex in columnsToDelete:
                    del self.dataFrame[columnIndex]
                    continue

                self.dataFrame[columnIndex] = self.dataFrame[columnIndex].map(dict(zip(unique, range(len(unique)))))

        self.dataFrame = self.dataFrame.fillna(self.dataFrame.median(axis=0))
        self.groundTruth = self.dataFrame[self.groundTruthColumns]

        # as requested in the assignment - "Please do not use the race and gender, and each of them can be the class"
        del self.dataFrame['race']
        del self.dataFrame['gender']
        super().reduceDimensions()


# ld = LoadDataSet2()
# ld.prepareDataset()
