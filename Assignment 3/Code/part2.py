# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt
import os
from ANNRegressor import ANNRegressor
import csv
import itertools

# Define error functions.
def meanSquaredError(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)

def stdSquaredError(yTrue, yPred):
    return np.std((yTrue - yPred) ** 2)

# Define file paths.
trainPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 3/dataset/train1.txt"
testPath = "/Users/aral/Documents/Bilkent Archive/GE 461 - Introduction to Data Science/Assignment 3/dataset/test1.txt"
outputDir = "ANN_Hyperparameter_Results"
os.makedirs(outputDir, exist_ok=True)

# Define various hyperparameter options.
hiddenUnitOptions = [2, 4, 8, 16, 32]
learningRateOptions = [0.01, 0.0005, 0.0001]
epochOptions = [1000]

# Load and normalize the dataset.
trainData = np.loadtxt(trainPath)
testData = np.loadtxt(testPath)

XTrain, yTrain = trainData[:, 0:1], trainData[:, 1:2]
XTest, yTest = testData[:, 0:1], testData[:, 1:2]

XMean = np.mean(XTrain)
XStd = np.std(XTrain)
XTrainNorm = (XTrain - XMean) / XStd
XTestNorm = (XTest - XMean) / XStd

# Try every possible combination of hyperparameters.
results = []

combinations = list(itertools.product(hiddenUnitOptions, learningRateOptions, epochOptions))

for hiddenUnits, learningRate, epochs in combinations:
    print(f"Training ANN with {hiddenUnits} units | LR = {learningRate} | Epochs = {epochs}")

    # Initialize and train.
    annModel = ANNRegressor(inputDim=1, hiddenDim=hiddenUnits)
    annModel.train(XTrainNorm, yTrain, learningRate=learningRate, epochs=epochs)

    # Evaluate the model.
    trainPreds = annModel.predict(XTrainNorm)
    testPreds = annModel.predict(XTestNorm)

    results.append({
        "Hidden Units": hiddenUnits,
        "Learning Rate": learningRate,
        "Epochs": epochs,
        "Train MSE": round(meanSquaredError(yTrain, trainPreds), 3), 
        "Train STD": round(stdSquaredError(yTrain, trainPreds), 3),
        "Test MSE": round(meanSquaredError(yTest, testPreds), 3),
        "Test STD": round(stdSquaredError(yTest, testPreds), 3)
    })

    # Plot predictions and hidden activations.
    xRange = np.linspace(XTrainNorm.min(), XTrainNorm.max(), 200).reshape(-1, 1)
    yCurve = annModel.predict(xRange)
    z1 = np.dot(xRange, annModel.weight1) + annModel.bias1
    a1 = annModel.sigmoid(z1)

    plt.figure(figsize=(10, 5))
    plt.title(f"ANN Output and Hidden Activations\nUnits={hiddenUnits}, LR={learningRate}, Epochs={epochs}")
    plt.scatter(XTrainNorm, yTrain, label="Training Data", color="blue", alpha=0.6)
    plt.plot(xRange, yCurve, label="ANN Output", color="red", linewidth=2)

    plt.xlabel("Normalized Input")
    plt.ylabel("Output / Activation")
    plt.legend(fontsize=8)
    plt.tight_layout()

    fileName = f"ann_h{hiddenUnits}_lr{learningRate}_ep{epochs}.png".replace('.', '_')
    plt.savefig(os.path.join(outputDir, fileName))
    plt.close()

# Export a csv table of results.
csvPath = os.path.join(outputDir, "grid_search_results.csv")
with open(csvPath, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

