import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Load the training and validation datasets
train_data = pd.read_csv('multi_train_set_pca.csv')
val_data = pd.read_csv('multi_validation_set_pca.csv')

# X as features and y as actual labels
X_train = train_data.drop(['quality_label'], axis=1)
y_train = train_data['quality_label']
X_val = val_data.drop(['quality_label'], axis=1)
y_val = val_data['quality_label']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define hyperparameter values for tuning
param_name = 'var_smoothing'
param_values = [1e-9, 1e-8, 1e-7]  # Variance smoothing parameter

print("\n=== Naïve Bayes Classifier ===")

# Cross-validation for hyperparameter tuning
cv_scores = []
for value in param_values:
    model = GaussianNB(var_smoothing=value)
    kf = KFold(n_splits=5, shuffle=True, random_state=25)
    cv_result = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    mean_score = np.mean(cv_result)
    cv_scores.append(mean_score)
    print(f"{param_name} = {value:.9f}, Accuracy = {mean_score:.4f}")  # Use .9f to display as decimal

# Plot cross-validation results
plt.figure(figsize=(8, 5))
plt.plot(param_values, cv_scores, marker='o')
plt.title('5-Fold Cross Validation Accuracy for Naïve Bayes Classifier')
plt.xlabel(param_name)
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('nb_multi_pca_cv_plot.png')

# Train best model (using the best parameter from CV)
best_param = param_values[np.argmax(cv_scores)]
model = GaussianNB(var_smoothing=best_param)

# Train, record time
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train

# Predict and assess accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

# Predict on validation set and record time
start_test = time.time()
val_predictions = model.predict(X_val)
end_test = time.time()
test_time = end_test - start_test

# Assess accuracy
val_accuracy = accuracy_score(y_val, val_predictions)

# Print results
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Training time: {train_time:.4f} seconds")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Testing time: {test_time:.4f} seconds")