import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP import load_data
from statsmodels.stats.anova import anova_lm

# Load dataset.
autoDataset = load_data("Auto")
print(autoDataset.info())

# Compute and visualize correlation matrix.
correlationMatrix = autoDataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlationMatrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Auto Dataset")
plt.show()

# Fit multiple linear regression model.
predictors = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin"]
formula = "mpg ~ " + " + ".join(predictors)
model = smf.ols(formula=formula, data=autoDataset).fit()
print("\nMultiple Linear Regression Results:\n", model.summary())

# Perform Analysis of Variance (ANOVA).
anovaResult = anova_lm(model)
print("\nANOVA Results:\n", anovaResult)

# Generate the Residual plot.
plt.figure(figsize=(8,6))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Generate the Influence plot.
fig, ax = plt.subplots(figsize=(8,6))
sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
plt.title("Influence Plot (Cook's Distance)")
plt.show()

# Add an interaction term defined as horsepower * acceleration.
autoDataset["horsepowerAcc"] = autoDataset["horsepower"] * autoDataset["acceleration"]
interactionFormula = formula + " + horsepowerAcc"
interactionModel = smf.ols(formula=interactionFormula, data=autoDataset).fit()
print("\nRegression with Interaction Term (horsepowerAcc):\n", interactionModel.summary())

# Apply the example transformations. 
autoDataset["log_weight"] = np.log(autoDataset["weight"])
autoDataset["sqrt_displacement"] = np.sqrt(autoDataset["displacement"])
formula_transformed = "mpg ~ cylinders + sqrt_displacement + horsepower + log_weight + acceleration + year + origin"
model_transformed = smf.ols(formula=formula_transformed, data=autoDataset).fit()
print("\nRegression with Transformed Variables:\n", model_transformed.summary())
