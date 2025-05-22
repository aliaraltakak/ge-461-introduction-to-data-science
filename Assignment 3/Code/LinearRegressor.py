# Import the required libraries.
import numpy as np
import os

# Define the Linear Regressor model.
class LinearRegressor:
    def __init__(self, inputDim):
        
        # Initialize weight and bias with small values.
        self.weight = np.random.randn(inputDim, 1) * 0.01
        self.bias = 0.0

    # Define the prediction method (forward pass).
    def predict(self, x):
        return np.dot(x, self.weight) + self.bias

    # Define the training process using stochastic gradient descent.
    def train(self, xTrain, yTrain, learningRate=0.01, epochs=100):
        sampleCount = xTrain.shape[0]
        epochLosses = []

        for epoch in range(epochs):
            totalLoss = 0

            for i in range(sampleCount):
                xi = xTrain[i:i+1]
                yi = yTrain[i:i+1]

                # Perform forward pass and compute loss.
                prediction = self.predict(xi)
                error = prediction - yi
                loss = np.sum(error ** 2)
                totalLoss += loss

                # Compute gradients.
                gradWeight = 2 * np.dot(xi.T, error)
                gradBias = 2 * error

                # Update weights and bias.
                self.weight -= learningRate * gradWeight
                self.bias -= learningRate * gradBias.item()

            # Compute average loss for the epoch.
            averageLoss = totalLoss / sampleCount
            epochLosses.append(averageLoss)

            # Print training statistics.
            print(f"[Linear] Epoch {epoch + 1:3d}: Loss = {averageLoss:.4f}")

        return epochLosses