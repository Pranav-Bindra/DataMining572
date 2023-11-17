import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the data
labels = pd.read_csv('label.csv', header=None)
data = pd.read_csv('data.csv', header=None)

# Convert data to NumPy array
data_array = data.values

# Number of clusters (K) is the number of unique labels in the labels dataset
K = len(labels[0].unique())

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# Function to calculate Cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Function to calculate Jaccard distance
def jaccard_distance(X, Y):
    numerator = np.sum(np.minimum(X, Y))
    denominator = np.sum(np.maximum(X, Y))
    
    jaccard_coefficient = numerator / denominator if denominator != 0 else 0
    jaccard_distance = 1 - jaccard_coefficient
    
    return jaccard_distance

# K-means algorithm with Euclidean distance
def kmeans_euclidean(data, K, max_iters=100):
    centroids = data[np.random.choice(range(len(data)), K, replace=False)]
    
    for _ in range(max_iters):
        assignments = np.argmin(np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data]), axis=1)
        centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
    
    return assignments, centroids

# K-means algorithm with Cosine similarity
def kmeans_cosine(data, K, max_iters=100):
    centroids = data[np.random.choice(range(len(data)), K, replace=False)]
    
    for _ in range(max_iters):
        assignments = np.argmax(np.array([[cosine_similarity(point, centroid) for centroid in centroids] for point in data]), axis=1)
        centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
    
    return assignments, centroids

# K-means algorithm with Jaccard distance
def kmeans_jaccard(data, K, max_iters=100):
    centroids = data[np.random.choice(range(len(data)), K, replace=False)]
    
    for _ in range(max_iters):
        distances = np.array([[jaccard_distance(point, centroid) for centroid in centroids] for point in data])
        assignments = np.argmin(distances, axis=1)
        new_centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
        centroids = new_centroids
    
    return assignments, centroids

# Function to assign cluster labels based on majority vote
def assign_cluster_labels(assignments, true_labels, K):
    cluster_labels = []
    for k in range(K):
        cluster_indices = np.where(assignments == k)[0]
        cluster_true_labels = true_labels.iloc[cluster_indices]
        majority_vote_label = cluster_true_labels.mode().iloc[0]
        cluster_labels.append(majority_vote_label)
    return cluster_labels

# Run K-means with Euclidean distance
assignments_euclidean, _ = kmeans_euclidean(data_array, K)
cluster_labels_euclidean = assign_cluster_labels(assignments_euclidean, labels, K)

# Run K-means with Cosine similarity
assignments_cosine, _ = kmeans_cosine(data_array, K)
cluster_labels_cosine = assign_cluster_labels(assignments_cosine, labels, K)

# Run K-means with Jaccard distance
assignments_jaccard, _ = kmeans_jaccard(data_array, K)
cluster_labels_jaccard = assign_cluster_labels(assignments_jaccard, labels, K)

# Calculate predictive accuracy
true_labels = labels.iloc[:, 0].values

accuracy_euclidean = accuracy_score(true_labels, [cluster_labels_euclidean[assignment] for assignment in assignments_euclidean])
accuracy_cosine = accuracy_score(true_labels, [cluster_labels_cosine[assignment] for assignment in assignments_cosine])
accuracy_jaccard = accuracy_score(true_labels, [cluster_labels_jaccard[assignment] for assignment in assignments_jaccard])

print(f"Accuracy for Euclidean-K-means: {accuracy_euclidean:.4f}")
print(f"Accuracy for Cosine-K-means: {accuracy_cosine:.4f}")
print(f"Accuracy for Jaccard-K-means: {accuracy_jaccard:.4f}")
