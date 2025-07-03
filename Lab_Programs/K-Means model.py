import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
cluster_labels = KMeans(n_clusters=4).fit_predict(X)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k')
plt.scatter(KMeans(n_clusters=4).fit(X).cluster_centers_[:, 0], KMeans(n_clusters=4).fit(X).cluster_centers_[:, 1], marker='o', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
