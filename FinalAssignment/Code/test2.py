import matplotlib.pyplot as plt

k = [1,2,3]
g = [2,3,1]
n_clusters_range = range(1, len(k)+1)
plt.plot(n_clusters_range, k, 'o-', label="k")
plt.plot(n_clusters_range, g, 'o-', label="g")
plt.show()