import numpy as np
import pandas as pd

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
def kmeans_euclidean(data, K, max_iters=500, tol=1e-4):
    centroids = data[np.random.choice(range(len(data)), K, replace=False)]
    
    for _ in range(max_iters):
        assignments = np.argmin(np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data]), axis=1)
        new_centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
        
        # Check for convergence
        if np.sum(np.abs(new_centroids - centroids)) < tol:
            break
        
        centroids = new_centroids
    
    sse = np.sum([euclidean_distance(data[i], centroids[assignments[i]])**2 for i in range(len(data))])
    
    return sse, _

# K-means algorithm with Cosine similarity
def kmeans_cosine(data, K, max_iters=500, tol=1e-4):
    centroids = data[np.random.choice(range(len(data)), K, replace=False)]
    
    for _ in range(max_iters):
        assignments = np.argmax(np.array([[cosine_similarity(point, centroid) for centroid in centroids] for point in data]), axis=1)
        new_centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
        
        # Check for convergence
        if np.sum(np.abs(new_centroids - centroids)) < tol:
            break
        
        centroids = new_centroids
    
    sse = np.sum([1 - cosine_similarity(data[i], centroids[assignments[i]]) for i in range(len(data))])
    
    return sse, _

# K-means algorithm with Jaccard distance
def kmeans_jaccard(data, K, max_iters=500, tol=1e-4):
    centroids = data[np.random.choice(range(len(data)), K, replace=False)]
    
    for _ in range(max_iters):
        distances = np.array([[jaccard_distance(point, centroid) for centroid in centroids] for point in data])
        assignments = np.argmin(distances, axis=1)
        new_centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
        
        # Check for convergence
        if np.sum(np.abs(new_centroids - centroids)) < tol:
            break
        
        centroids = new_centroids
    
    sse = np.sum(distances[np.arange(len(distances)), assignments])
    
    return sse, _

# Run K-means with Euclidean distance
sse_euclidean, iterations_euclidean = kmeans_euclidean(data_array, K)
print(f"SSE for Euclidean-K-means: {sse_euclidean}, Iterations: {iterations_euclidean}")

# Run K-means with Cosine similarity
sse_cosine, iterations_cosine = kmeans_cosine(data_array, K)
print(f"SSE for Cosine-K-means: {sse_cosine}, Iterations: {iterations_cosine}")

# Run K-means with Jaccard distance
sse_jaccard, iterations_jaccard = kmeans_jaccard(data_array, K)
print(f"SSE for Jaccard-K-means: {sse_jaccard}, Iterations: {iterations_jaccard}")
