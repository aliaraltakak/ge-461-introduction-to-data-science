# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import multivariate_normal

# Define a function to train the Gaussian classifier.
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

# Define a vectorized function to make predictions with the Gaussian classifier.
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

# Set random seed.
studentID = 22001758
np.random.seed(studentID)

# Load data and labels.
fashionDataPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 2/fashion_mnist/fashion_mnist_data.txt"
fashionLabelsPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 2/fashion_mnist/fashion_mnist_labels.txt"

fashionData = np.loadtxt(fashionDataPath)
fashionLabels = np.loadtxt(fashionLabelsPath)

# Split the dataset.
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

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Compute the overall mean from the entire dataset and center the data.
meanWholeData = np.mean(fashionData, axis=0)
X_train_centered = X_train - meanWholeData
X_test_centered = X_test - meanWholeData

# Apply LDA on the training data.
ldaModel = LinearDiscriminantAnalysis(n_components=9, solver='eigen')
ldaModel.fit(X_train_centered, y_train)

# Display the LDA bases as images.
numLdaComponents = 9
plt.figure(figsize=(12, 8))
for i in range(numLdaComponents):
    plt.subplot(3, 3, i + 1)
    basisImage = ldaModel.scalings_[i]
    plt.imshow(basisImage.reshape(28, 28))
    plt.title(f"LDA Basis {i+1}")
    plt.axis('off')
plt.suptitle("LDA Bases as Visualized")
plt.show()


# Evaluate the classifier on different LDA subspace dimensions.
dimensions = range(1, 10)
trainErrorsLDA = []
testErrorsLDA = []

for dimension in dimensions:
    # Project data onto the first d LDA components.
    X_train_lda = ldaModel.transform(X_train_centered)[:, :dimension]
    X_test_lda = ldaModel.transform(X_test_centered)[:, :dimension]
    
    # Train and evaluate the Gaussian classifier.
    params = trainGaussianClassifier(X_train_lda, y_train)
    y_train_pred = predictGaussianClassifier(X_train_lda, params)
    y_test_pred = predictGaussianClassifier(X_test_lda, params)
    
    trainError = np.mean(y_train_pred != y_train)
    testError = np.mean(y_test_pred != y_test)
    
    trainErrorsLDA.append(trainError)
    testErrorsLDA.append(testError)
    
    print(f"Dimension: {dimension}, Train Error: {trainError:.4f}, Test Error: {testError:.4f}")

# Plot the classification errors vs. LDA subspace dimensions.
plt.figure()
plt.plot(dimensions, trainErrorsLDA, marker='o', label='Training Error')
plt.xlabel("Number of LDA Components")
plt.ylabel("Classification Error")
plt.title("Training Error vs. LDA Subspace Dimension")
plt.legend()
plt.show()

plt.figure()
plt.plot(dimensions, testErrorsLDA, marker='o', color='orange', label='Test Error')
plt.xlabel("Number of LDA Components")
plt.ylabel("Classification Error")
plt.title("Test Error vs. LDA Subspace Dimension")
plt.legend()
plt.show()