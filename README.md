# GE 461: Introduction to Data Science – Spring 2024–2025

## Project Repository

This repository contains solutions to five projects assigned in GE 461 – Introduction to Data Science at Bilkent University in the Spring 2025 semester.

---

## Project 1: Linear Regression Analysis

### Overview

This project focuses on exploring linear regression techniques using the `Auto` dataset. The project is based on exercises 3.7.8 and 3.7.9 from the book *"An Introduction to Statistical Learning with Applications in Python and R."*

### Tasks

- Implemented simple linear regression using `horsepower` to predict `mpg`.
- Analyzed coefficient signs, p-values, and R² value to interpret relationships.
- Predicted `mpg` for given values and computed 95% confidence/prediction intervals.
- Extended analysis to multiple linear regression with several predictors.
- Conducted:
  - Correlation analysis
  - Residual and leverage plot inspection
  - Interaction term analysis
  - Variable transformation analysis (log, square root, polynomial)

---

## Project 2: Dimensionality Reduction and Visualization

### Dataset

Subset of 10,000 Fashion-MNIST samples (1,000 per class), each as a 784-dimensional vector.

### Tasks

#### Part 1: Principal Component Analysis (PCA)

- Centered training data and computed eigenvalues of the covariance matrix.
- Plotted eigenvalue spectrum and selected principal components.
- Visualized sample mean and top eigenvectors.
- Trained a Gaussian classifier with varying dimensions (1–400) and plotted classification errors.

#### Part 2: Linear Discriminant Analysis (LDA)

- Projected data to 1–9 dimensional subspaces using LDA.
- Trained and evaluated Gaussian classifier.
- Compared LDA and PCA in terms of classification accuracy and dimensionality.

#### Part 3: t-SNE Visualization

- Projected dataset into 2D using t-SNE.
- Plotted class-labeled data.
- Discussed perplexity, iterations, and clustering performance.

---

## Project 3: Artificial Neural Network for Regression

### Overview

Built ANN regressors using backpropagation, including:

- Linear regression (no hidden layer)
- Single-hidden-layer ANN with sigmoid activation

### Tasks

#### Part A: Configuration Tuning

- Tuned learning rate, weight initialization, number of epochs, and normalization.
- Reported configuration and model performance.

#### Part B: Output Visualization

- Plotted actual vs. predicted outputs for both training and test sets.
- Used dense sampling to draw smooth predicted output curves.

#### Part C: Complexity Analysis

- Trained models with 2, 4, 8, 16, 32 hidden units.
- Plotted per-unit hidden activations and outputs.
- Reported losses with standard deviation in a summary table.
- Analyzed the effect of ANN complexity on generalization.

---

## Project 4: Fall Detection from Wearable Sensors

### Dataset

566 samples, 306 sensor-based features, labeled:
- `F`: Fall
- `NF`: Non-fall

### Tasks

#### Part A: Clustering and PCA

- Applied PCA to reduce features to 2D.
- Performed k-means clustering with varying `k`.
- Compared clusters to original labels and evaluated clustering validity.

#### Part B: Supervised Classification

- Used 70/15/15 split for training/validation/test.
- Implemented and evaluated:
  - Support Vector Machine (SVM)
  - Multi-layer Perceptron (MLP)
- Tuned hyperparameters and reported comparative accuracy.

---

## Project 5: Data Stream Mining and Concept Drift

### Objective

Develop evolving models for data stream classification with concept drift using `scikit-multiflow`.

### Tasks

#### Datasets

- **Synthetic**: AGRAWAL, SEA (100,000 samples with drift at 35k and 60k)
- **Real**: Spam, Electricity datasets

#### Models

- Adaptive Random Forest (ARF)
- SAM-kNN

#### Drift Detection

- Combined base learners with DDM, EDDM, ADWIN detectors
- Implemented adaptive weighting and ensemble voting
- Developed **Passive Drift Detection** strategy (no external detectors)

#### Evaluation

- Prequential (test-then-train) evaluation
- Accuracy plots using sliding windows
- Compared:
  - Ensemble vs. standalone models
  - Passive vs. active drift detection
  - Window sizes
- Analyzed reaction speed, false alarms, noise tolerance, and accuracy stability
