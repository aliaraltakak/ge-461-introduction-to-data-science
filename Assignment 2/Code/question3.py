# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set random seed.
studentID = 22001758
np.random.seed(studentID)

# Load data and labels.
fashionDataPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 2/fashion_mnist/fashion_mnist_data.txt"
fashionLabelsPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 2/fashion_mnist/fashion_mnist_labels.txt"

fashionData = np.loadtxt(fashionDataPath)
fashionLabels = np.loadtxt(fashionLabelsPath)

# Compute the overall mean and center the data.
meanWholeData = np.mean(fashionData, axis=0)
fashionData_centered = fashionData - meanWholeData

# Define the parameter ranges.
perplexityValues = [5, 10, 15, 20, 25, 30]
iterationCount = [250, 500, 750, 1000]

# Loop over each combination and generate an individual plot.
for n_iter in iterationCount:
    for perplexity in perplexityValues:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=studentID)
        tsne_mapping = tsne.fit_transform(fashionData_centered)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_mapping[:, 0], tsne_mapping[:, 1],
                    c=fashionLabels, cmap='tab10', s=1, alpha=0.7)
        plt.title(f"t-SNE Mapping (Perplexity: {perplexity}, n_iter: {n_iter})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(ticks=range(10), label="Class")
        plt.show()
