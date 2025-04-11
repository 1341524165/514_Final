import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

#  Load the training dataset
# data = pd.read_csv('binary_train_set.csv')
data = pd.read_csv('binary_train_set_pca.csv')

# X as features and y as actual labels
X = data.drop(['quality_label'], axis=1)
y = data['quality_label']

# ----------------------model & hyperparameter----------------------
# # 1. k-NN
# from sklearn.neighbors import KNeighborsClassifier
# model_name = 'k-NN'
# param_name = 'n_neighbors'
# param_values = [3, 5, 7, 9, 11]

# 2. Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_name = 'Decision Tree'
param_name = 'max_depth'
param_values = [3, 5, 7, 9, 11]

# def a func for changing diff models
def create_model(value):
    # return KNeighborsClassifier(n_neighbors=value)
    return DecisionTreeClassifier(max_depth=value)
# ----------------------model & hyperparameter----------------------

# store cross-validation scores
cv_scores = []

for value in param_values:
    model = create_model(value)
    kf = KFold(n_splits=5, shuffle=True, random_state=25)
    cv_result = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    mean_score = np.mean(cv_result)
    cv_scores.append(mean_score)
    print(f"{param_name} = {value}, Accuracy = {mean_score:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(param_values, cv_scores, marker='o')
plt.title(f'5-Fold Cross Validation Accuracy for {model_name}')
plt.xlabel(param_name)
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
