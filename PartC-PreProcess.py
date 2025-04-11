import pandas as pd
from sklearn.model_selection import train_test_split

# data = pd.read_csv('binary_classification.csv')
data = pd.read_csv('multi_classification.csv')

# X as features and y as actual labels
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=25, shuffle=True)

# Print dimensions to confirm
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# Save to CSV files
train_set = X_train.copy()
train_set['quality_label'] = y_train
# train_set.to_csv('binary_train_set.csv', index=False)
train_set.to_csv('multi_train_set.csv', index=False)
val_set = X_val.copy()
val_set['quality_label'] = y_val
# val_set.to_csv('binary_validation_set.csv', index=False)
val_set.to_csv('multi_validation_set.csv', index=False)

# print("'binary_train_set.csv' and 'binary_validation_set.csv' saved")
print("'multi_train_set.csv' and 'multi_validation_set.csv' saved")