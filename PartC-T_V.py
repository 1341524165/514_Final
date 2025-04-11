import pandas as pd
import time
from sklearn.metrics import accuracy_score

# Load the training and validation datasets
# train_data = pd.read_csv('binary_train_set.csv')
# val_data = pd.read_csv('binary_validation_set.csv')
train_data = pd.read_csv('binary_train_set_pca.csv')
val_data = pd.read_csv('binary_validation_set_pca.csv')

# X as features and y as actual labels
X_train = train_data.drop(['quality_label'], axis=1)
y_train = train_data['quality_label']
X_val = val_data.drop(['quality_label'], axis=1)
y_val = val_data['quality_label']

# ----------------------model----------------------
# 1. k-NN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

# # 2. Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(max_depth=7)
# ----------------------model----------------------

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
# assess accuracy
val_accuracy = accuracy_score(y_val, val_predictions)


print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Training time: {train_time:.4f} seconds")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Testing time: {test_time:.4f} seconds")
