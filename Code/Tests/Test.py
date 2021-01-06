import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
random_state = 1

# load the data
data = pd.read_csv("C:\\UnsupervisedLearning\\MidTermAssignment\\dataset\\Dataset3\\e-shop clothing 2008.csv",sep=";",nrows=100)
del data["page 2 (clothing model)"]

# GMM
gmm = GaussianMixture(n_components=4,random_state=random_state)
gmm.fit(data)
labels = gmm.predict(data)

frame = pd.DataFrame(data)

frame = StandardScaler().fit_transform(frame)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(frame)
frame = pd.DataFrame(data=principalComponents, columns=['dim1','dim2'])

frame['cluster'] = labels


color=['blue','green','cyan', 'black']

dims = ['dim1','dim2']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["dim1"],data["dim2"],c=color[k])

plt.show()