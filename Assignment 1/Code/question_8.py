# Import the libraries. 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from ISLP import load_data
import pandas as pd
import numpy as np
import ISLP.models

# Define the function to draw the regression line.
def abline(ax, b, m):
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, 'r', linewidth=2)

# Load the dataset.
autoDataset = load_data("Auto")

# Initially, create a model matrix.
X = pd.DataFrame({ "intercept": np.ones(autoDataset.shape[0]),
                    "horsepower": autoDataset["horsepower"]})

# Extract the response.
y = autoDataset["mpg"]

# Define and fit the simple linear regression model.
model = sm.OLS(y, X)
results = model.fit()

# Visualize the results.
print("\n")
print(ISLP.models.summarize(results))
print("\n")

# Extract the intercept and slope from the regression results.
intercept = results.params['intercept']
slope = results.params['horsepower']

# Plot the data points.
fig, ax = plt.subplots()
ax.scatter(autoDataset['horsepower'], autoDataset['mpg'], alpha=0.5)
ax.set_xlabel("Horsepower")
ax.set_ylabel("MPG")
ax.set_title("Regression of MPG on Horsepower")
plt.grid()

# Call the abline function with the regression parameters.
abline(ax, intercept, slope)

# Show the plot.
plt.show()

# Make mpg prediction for 98 HP.
horsePower98 = np.array([1, 98])  # Include intercept term
horsePowerPrediction = results.get_prediction(horsePower98)
summary_frame = horsePowerPrediction.summary_frame(alpha=0.05)  # 95% confidence level

# Extract the prediction values.
predictedMPG = summary_frame['mean'][0]
confidenceInterval = (summary_frame['mean_ci_lower'][0], summary_frame['mean_ci_upper'][0])
predictionInterval = (summary_frame['obs_ci_lower'][0], summary_frame['obs_ci_upper'][0])

# Print the results
print("\nPrediction for Horsepower = 98")
print(f"Predicted MPG: {predictedMPG:.2f}")
print(f"95% Confidence Interval: {confidenceInterval}")
print(f"95% Prediction Interval: {predictionInterval}")
print("\n")

# Determine the R^2 value for parameter effect strength.
r_squared = results.rsquared
print(f"R-Squared value for determination parameters: {r_squared:.4f}")
print("\n")


