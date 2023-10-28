import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the MNIST data
url = "http://yann.lecun.com/exdb/mnist/"
# Download the mnist_train.csv file and save it locally
# Read the file and extract the necessary columns

def mykmeans(digits, K, N):
    best_loss = float('inf')
    best_params = None
    best_assignments = None

    for _ in range(N):
        # Initialize clusters randomly
        initial_means = digits[np.random.choice(digits.shape[0], K, replace=False)]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=K, init=initial_means, n_init=1)
        kmeans.fit(digits)

        # Calculate loss
        loss = kmeans.inertia_

        if loss < best_loss:
            best_loss = loss
            best_params = kmeans.cluster_centers_
            best_assignments = kmeans.labels_

    return best_params, best_assignments, best_loss

# Define K values
K_values = [5, 10, 15, 20, 25]

# Initialize number of initializations N
N = 3

losses = []

for K in K_values:
    cluster_params, cluster_assignments, loss = mykmeans(digits, K, N)
    losses.append(loss)
    print(f"For K={K}, Loss={loss}")

# Plotting
plt.plot(K_values, losses, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Loss')
for i, loss in enumerate(losses):
    plt.text(K_values[i], loss, f'{loss:.2f}', ha='right', va='bottom')
plt.title('K-means Clustering Loss')
plt.show()
