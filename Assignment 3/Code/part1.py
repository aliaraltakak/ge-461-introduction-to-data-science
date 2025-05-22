# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt
import os
from LinearRegressor import LinearRegressor
from ANNRegressor import ANNRegressor

## Comment: The classes ANNRegressor and LinearRegressor are basically the same, however
## they differ as having a hidden layer and no layer. I defined them on different files 
## for the ease of reading and tracking.

# Define the mean squared error function.
def meanSquaredError(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)

# Define variables for datapath and configuration settings.
trainPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 3/dataset/train1.txt"
testPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 3/dataset/test1.txt"
outputDir = "Regressor_Comparison_Results"
os.makedirs(outputDir, exist_ok=True)
hiddenUnits = 64
learningRate = 0.001   
epochs = 1000

# Load and preprocess the dataset.
trainData = np.loadtxt(trainPath)
testData = np.loadtxt(testPath)

# Split into features and labels.
XTrain, yTrain = trainData[:, 0:1], trainData[:, 1:2]
XTest, yTest = testData[:, 0:1], testData[:, 1:2]

# Normalize the input features using training set statistics.
XMean = np.mean(XTrain)
XStd = np.std(XTrain)
XTrainNorm = (XTrain - XMean) / XStd
XTestNorm = (XTest - XMean) / XStd

# Train the Linear Regressor.
linearModel = LinearRegressor(inputDim=1)
linearLosses = linearModel.train(XTrainNorm, yTrain, learningRate=learningRate, epochs=epochs)

# Train the ANN Regressor.
annModel = ANNRegressor(inputDim=1, hiddenDim=hiddenUnits)
annLosses = annModel.train(XTrainNorm, yTrain, learningRate=learningRate, epochs=epochs)

# Get predictions from the linear model.
trainPredsLinear = linearModel.predict(XTrainNorm)
testPredsLinear = linearModel.predict(XTestNorm)

# Get predictions from the ANN model.
trainPredsANN = annModel.predict(XTrainNorm)
testPredsANN = annModel.predict(XTestNorm)

# Compute MSE for linear model.
trainMSELinear = meanSquaredError(yTrain, trainPredsLinear)
testMSELinear = meanSquaredError(yTest, testPredsLinear)

# Compute MSE for ANN.
trainMSEANN = meanSquaredError(yTrain, trainPredsANN)
testMSEANN = meanSquaredError(yTest, testPredsANN)

# Print model performance results.
print(f"\n[Linear] Train MSE: {trainMSELinear:.4f} | Test MSE: {testMSELinear:.4f}")
print(f"[ANN   ] Train MSE: {trainMSEANN:.4f} | Test MSE: {testMSEANN:.4f}")


# Visualize the model predictions.
xRange = np.linspace(XTrainNorm.min(), XTrainNorm.max(), 200).reshape(-1, 1)
yCurveLinear = linearModel.predict(xRange)
yCurveANN = annModel.predict(xRange)

# Initialize figure layout.
plt.figure(figsize=(12, 6))

# Plot for training set.
plt.subplot(1, 2, 1)
plt.title("Training Set")
plt.scatter(XTrainNorm, yTrain, label="Actual", color="blue")
plt.plot(xRange, yCurveLinear, label="Linear", color="orange")
plt.plot(xRange, yCurveANN, label="ANN", color="red")
plt.xlabel("Normalized Input")
plt.ylabel("Output")
plt.legend()

# Plot for test set.
plt.subplot(1, 2, 2)
plt.title("Test Set")
plt.scatter(XTestNorm, yTest, label="Actual", color="green")
plt.plot(xRange, yCurveLinear, label="Linear", color="orange")
plt.plot(xRange, yCurveANN, label="ANN", color="red")
plt.xlabel("Normalized Input")
plt.ylabel("Output")
plt.legend()

# Save the figure.
plt.tight_layout()
plotName = f"combined_hidden{hiddenUnits}_lr{learningRate}_ep{epochs}.png".replace('.', '_')
plt.savefig(os.path.join(outputDir, plotName))
plt.show()

# Plot the losses.
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), linearLosses, label="Linear Regressor", color="orange")
plt.plot(range(1, epochs + 1), annLosses, label="ANN Regressor", color="red")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True)

# Save the training loss plot.
lossPlotName = f"loss_hidden{hiddenUnits}_lr{learningRate}_ep{epochs}.png".replace('.', '_')
plt.savefig(os.path.join(outputDir, lossPlotName))
plt.show()

