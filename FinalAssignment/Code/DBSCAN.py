from sklearn.cluster import DBSCAN
import numpy as np
from Dataset2 import Dataset2


dataset = Dataset2()
df = dataset.get_data_frame()


dbscan = DBSCAN(eps=0.3, min_samples=5).fit(df)

labels = dbscan.labels_

print(list(labels))
print(df.iloc[labels])