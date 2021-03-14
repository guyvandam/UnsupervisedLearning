import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from Dataset2 import Dataset2


clf = LocalOutlierFactor()
ds = Dataset2()
ds.prepareDataset()

df = ds.get_data_frame()
results = clf.fit_predict(df)

print(list(results))
results = (results + 1) / 2
results = results.astype(bool)
results = np.invert(results)


print(df.iloc[results])