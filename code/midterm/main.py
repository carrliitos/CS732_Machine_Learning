from sklearn.cluster import KMeans

# Define the data points
data = [(1,1), (3,4), (2,3), (5,5), (7,9), (3,8), (8,0)]

# Initialize the K-means clustering algorithm with k=3
kmeans = KMeans(n_clusters=3, init=[[1,1],[5,5],[7,9]], n_init=1, random_state=0)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster centers
centroids = kmeans.cluster_centers_

# Get the labels assigned to each data point
labels = kmeans.labels_

# Print the clusters and centroids
for i in range(3):
    cluster = [data[j] for j in range(len(data)) if labels[j] == i]
    print(f'Cluster {i+1}: {cluster}')
    print(f'Centroid: {centroids[i]}')
