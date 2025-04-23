import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans 
# Generating synthetic data 
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, 
random_state=42) 
# Initialize K-Means with the number of clusters 
kmeans = KMeans(n_clusters=4) 
# Fit the K-Means model to the data 
kmeans.fit(X) 
# Predict cluster labels 
cluster_labels = kmeans.predict(X) 
# Visualize the clusters 
plt.figure(figsize=(7,5)) 
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
marker='o', s=200, color='red', label='Centroids') 
plt.title('K-Means Clustering') 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.legend() 
plt.show()
