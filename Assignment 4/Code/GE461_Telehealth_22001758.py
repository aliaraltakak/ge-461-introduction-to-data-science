# Import the required libraries.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset.
data = pd.read_csv('falldetection_dataset.csv', header=None)  
data = data.drop(0, axis=1)                                  
data = data.replace('F', 1)                                  
data = data.replace('NF', 0)                                 
labels = data[1]                                           
cl = ['NF', 'F']                                           
data = data.drop(1, axis=1)                                  
print(data.head())

# Standardize the features of the dataset.
scaler = StandardScaler()
scaledFeatures = scaler.fit_transform(data)

# Apply PCA and plot the explained variance ratio versus principal components.
pca_full = PCA()
PCAresult_full = pca_full.fit_transform(scaledFeatures)
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1), 
         np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio versus Principal Components')
plt.grid()
plt.show()

# Apply PCA again with only two components.
pca = PCA(n_components=2)
PCAresult = pca.fit_transform(scaledFeatures)
print("Explained Variance Ratio for PC1 and PC2:", round(pca.explained_variance_ratio_[0],3), round(pca.explained_variance_ratio_[1],3))

# Plot the projection of the first two components classified with their labels.
plt.figure(figsize=(8,6))
for i in range(2):
    ind = np.where(labels == i)[0] 
    plt.plot(PCAresult[ind, 0], PCAresult[ind, 1], 'o', label=cl[i], markersize=3)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scatter plot of data projected to two dimensions using PCA.")
plt.legend()
plt.show()

# Apply K-means clustering on the first two components.
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=22001758)
    clusters = kmeans.fit_predict(PCAresult)
    plt.figure(figsize=(8,6))
    plt.scatter(PCAresult[:, 0], PCAresult[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f'K-means Clustering (k={k})')
    plt.show()

# Run k-means clustering with k=2.
kmeans_2 = KMeans(n_clusters=2, random_state=22001758)
clusters_2 = kmeans_2.fit_predict(PCAresult)
comparison_df = pd.DataFrame({'Cluster': clusters_2, 'Action Label': labels})
print(comparison_df.groupby(['Cluster', 'Action Label']).size())

# --- Part B: Supervised Learning ---

encodedLabels = labels.values  

# Split the data into training (70%), validation (15%), and testing (15%) sets.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    scaledFeatures, encodedLabels, test_size=0.30, random_state=22001758, stratify=encodedLabels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=22001758, stratify=y_train_val
)
print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Apply the support vector machine classifier.
bestValAccuracy = 0
bestSVM = None
print("\nSVM Hyperparameter Tuning:")
for C in [0.1, 0.5, 1, 2, 5, 10]:
    for kernel in ['linear', 'rbf', 'poly']:
        svm_model = SVC(C=C, kernel=kernel, random_state=22001758)
        svm_model.fit(X_train, y_train)
        val_preds = svm_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"SVM (kernel={kernel}, C={C}) - Validation Accuracy: {val_acc:.4f}")
        if val_acc > bestValAccuracy:
            bestValAccuracy = val_acc
            bestSVM = svm_model

# Evaluate the best SVM on the test set.
SVMTestPredictions = bestSVM.predict(X_test)
SVMTestAccuracy = accuracy_score(y_test, SVMTestPredictions)
print(f"\nBest SVM classifier test accuracy: {SVMTestAccuracy:.4f}")
print("SVM Classification Report:")
print(classification_report(y_test, SVMTestPredictions, target_names=cl))
print("\nBest SVM parameters:")
print(bestSVM.get_params())

# Run the Multilayer Perceptron training algorithm.
bestValAccuracyMLP = 0
bestMLP = None
print("\nMLP Hyperparameter Tuning:")
for hidden_layer_sizes in [(50,), (75,), (100,), (125,), (50,50), (75,75), (100,100), (125,125)]:
    for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                                  max_iter= 1000, random_state=22001758)
        mlp_model.fit(X_train, y_train)
        val_preds = mlp_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"MLP (hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}) - Validation Accuracy: {val_acc:.4f}")
        if val_acc > bestValAccuracyMLP:
            bestValAccuracyMLP = val_acc
            bestMLP = mlp_model

# Evaluate the best MLP on the test set.
MLPTestPredictions = bestMLP.predict(X_test)
MLPTestAccuracy = accuracy_score(y_test, MLPTestPredictions)
print(f"\nBest MLP classifier test accuracy: {MLPTestAccuracy:.4f}")
print("MLP Classification Report:")
print(classification_report(y_test, MLPTestPredictions, target_names=cl))
print("\nBest MLP parameters:")
print(bestMLP.get_params())
