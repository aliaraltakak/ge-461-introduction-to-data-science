import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

# Set random seed.
studentID = 22001758
np.random.seed(studentID)

# Load data and labels.
fashionDataPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 2/fashion_mnist/fashion_mnist_data.txt"
fashionLabelsPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 2/fashion_mnist/fashion_mnist_labels.txt"

fashionData = np.loadtxt(fashionDataPath)
fashionLabels = np.loadtxt(fashionLabelsPath)

uniqueClasses = np.unique(fashionLabels)
trainIndices = []
testIndices = []

for c in uniqueClasses:
    indices = np.where(fashionLabels == c)[0]
    np.random.shuffle(indices)
    splitPoint = len(indices) // 2
    trainIndices.extend(indices[:splitPoint])
    testIndices.extend(indices[splitPoint:])

trainIndices = np.array(trainIndices)
testIndices = np.array(testIndices)

X_train = fashionData[trainIndices, :]
y_train = fashionLabels[trainIndices]
X_test = fashionData[testIndices, :]
y_test = fashionLabels[testIndices]

print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

# Compute the overall mean of the entire dataset.
meanWholeData = np.mean(fashionData, axis=0)

# Display the mean image before subtracting the mean.
plt.figure()
plt.imshow(meanWholeData.reshape(28, 28), cmap='gray')
plt.title("Mean Image of Dataset")
plt.axis('off')
plt.show()

# Center both the training and the test sets.
X_train_centered = X_train - meanWholeData
X_test_centered = X_test - meanWholeData

# Compute PCA on the training data (up to 400 components).
numComponentsMax = 400
pcaModel = PCA(n_components=numComponentsMax)
pcaModel.fit(X_train_centered)

# Extract eigenvalues (explained variance) and plot the scree plot.
eigenValues = pcaModel.explained_variance_
plt.figure()
plt.plot(np.arange(1, numComponentsMax + 1), eigenValues)
plt.title("Scree Plot of PCA Eigenvalues")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.show()

# Display the top 10 eigenvectors as images.
numEigenvectorsToShow = 10
plt.figure(figsize=(12, 6))
for i in range(numEigenvectorsToShow):
    plt.subplot(2, 5, i + 1)
    plt.imshow(pcaModel.components_[i].reshape(28, 28), cmap='gray')
    plt.title(f"Eigenvector {i+1}")
    plt.axis('off')
plt.suptitle("Top 10 PCA Eigenvectors")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Define a function to train a Gaussian classifier.
def trainGaussianClassifier(data, labels):
    classParams = {}
    for c in np.unique(labels):
        data_c = data[labels == c]
        mu = np.mean(data_c, axis=0)
        sigma = np.cov(data_c, rowvar=False, bias=True)
        sigma = np.atleast_2d(sigma)
        sigma += 1e-6 * np.eye(sigma.shape[0])
        classParams[c] = (mu, sigma)
    return classParams

# Define a function to make predictions with the Gaussian classifier.
def predictGaussianClassifier(data, classParams):

    nSamples, d = data.shape
    classes = sorted(classParams.keys())
    logLikesMatrix = np.zeros((nSamples, len(classes)))
    
    for idx, c in enumerate(classes):
        mu, sigma = classParams[c]
        sigmaInv = np.linalg.inv(sigma)
        sign, logDetSigma = np.linalg.slogdet(sigma)
        diff = data - mu  
        mahalanobis = np.sum(diff @ sigmaInv * diff, axis=1)
        logLikesMatrix[:, idx] = -0.5 * mahalanobis - 0.5 * d * np.log(2 * np.pi) - 0.5 * logDetSigma

    bestClassIndices = np.argmax(logLikesMatrix, axis=1)
    y_pred = np.array([classes[i] for i in bestClassIndices])
    return y_pred

# List of PCA subspace dimensions to evaluate.
dimensionList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15,
                 20, 25, 30, 35, 40, 45, 50, 55, 60,
                 65, 70, 75, 80, 85, 90, 95, 100, 125,
                 150, 175, 200, 225, 250, 275, 300, 325, 
                 350, 375, 400]

trainErrors = []
testErrors = []

for dimension in dimensionList:
    # Project onto the first d principal components.
    W = pcaModel.components_[:dimension].T 
    X_train_proj = X_train_centered @ W
    X_test_proj = X_test_centered @ W
    
    # Train the Gaussian classifier on the projected data.
    params = trainGaussianClassifier(X_train_proj, y_train)
    
    # Use the vectorized prediction function.
    y_train_pred = predictGaussianClassifier(X_train_proj, params)
    y_test_pred = predictGaussianClassifier(X_test_proj, params)
    
    # Compute classification error.
    trainError = np.mean(y_train_pred != y_train)
    testError = np.mean(y_test_pred != y_test)
    trainErrors.append(trainError)
    testErrors.append(testError)
    
    print(f"Components: {dimension}, Train Error: {trainError:.4f}, Test Error: {testError:.4f}")

# Plot training error versus number of PCA components.
plt.figure()
plt.plot(dimensionList, trainErrors, label='Training Error')
plt.xlabel("Number of PCA Components")
plt.ylabel("Classification Error")
plt.title("Training Error vs. PCA Subspace Dimension")
plt.legend()
plt.show()

# Plot test error versus number of PCA components.
plt.figure()
plt.plot(dimensionList, testErrors, color='orange', label='Test Error')
plt.xlabel("Number of PCA Components")
plt.ylabel("Classification Error")
plt.title("Test Error vs. PCA Subspace Dimension")
plt.legend()
plt.show()
