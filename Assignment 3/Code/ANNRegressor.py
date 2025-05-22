# Import the required libraries.
import numpy as np
import os

# Define the ANN Regressor model with a single hidden layer.
class ANNRegressor:
    def __init__(self, inputDim, hiddenDim):

        # Initialize weights and biases for both layers.
        self.weight1 = np.random.randn(inputDim, hiddenDim) * 0.1
        self.bias1 = np.zeros((1, hiddenDim))
        self.weight2 = np.random.randn(hiddenDim, 1) * 0.1
        self.bias2 = 0.0

    # Define the sigmoid activation function.
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Define the derivative of the sigmoid function.
    def sigmoidDerivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    # Define the prediction method (forward pass).
    def predict(self, x):
        z1 = np.dot(x, self.weight1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.weight2) + self.bias2
        return z2

    # Define the training process using backpropagation with SGD.
    def train(self, xTrain, yTrain, learningRate=0.01, epochs=100):
        sampleCount = xTrain.shape[0]
        epochLosses = []

        for epoch in range(epochs):
            totalLoss = 0

            for i in range(sampleCount):
                xi = xTrain[i:i+1]
                yi = yTrain[i:i+1]

                # Execute forward pass.
                z1 = np.dot(xi, self.weight1) + self.bias1
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.weight2) + self.bias2
                prediction = z2

                # Compute loss.
                error = prediction - yi
                loss = np.sum(error ** 2)
                totalLoss += loss

                # Execute backward pass.
                dZ2 = 2 * error
                dW2 = np.dot(a1.T, dZ2)
                dB2 = dZ2.item()
                dA1 = np.dot(dZ2, self.weight2.T)
                dZ1 = dA1 * self.sigmoidDerivative(z1)
                dW1 = np.dot(xi.T, dZ1)
                dB1 = dZ1

                # Update weights and biases.
                self.weight2 -= learningRate * dW2
                self.bias2 -= learningRate * dB2
                self.weight1 -= learningRate * dW1
                self.bias1 -= learningRate * dB1

            # Compute average loss for this epoch.
            averageLoss = totalLoss / sampleCount
            epochLosses.append(averageLoss)

            # Print training statistics.
            print(f"[ANN] Epoch {epoch + 1:3d}: Loss = {averageLoss:.4f}")

        return epochLosses