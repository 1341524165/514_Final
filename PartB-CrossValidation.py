import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

data = pd.read_csv('binary_classification.csv')

# X as features and y as actual labels
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

k_values = [3, 5, 7, 9, 11]

# store cross-validation scores
cv_scores = []

# -------------------cross-validation-------------------
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=25)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    mean_score = np.mean(scores)
    cv_scores.append(mean_score)
    print(f"k = {k}, Accuracy = {mean_score:.4f}")



plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o')
plt.title('5-Fold Cross Validation Accuracy for k-NN')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()
