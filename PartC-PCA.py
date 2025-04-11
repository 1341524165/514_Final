import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
# data = pd.read_csv('binary_classification.csv')
data = pd.read_csv('multi_classification.csv')

# X as features and y as actual labels
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# Standardization
X_scaled = StandardScaler().fit_transform(X)
# PCA => 5 components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Combine PCA results with labels
pca_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
pca_data['quality_label'] = y

# pca_data.to_csv('binary_classification_pca.csv', index=False)
# print("'binary_classification_pca.csv' saved")
pca_data.to_csv('multi_classification_pca.csv', index=False)
print("'multi_classification_pca.csv' saved")



# --------------------------------Use the PreProcess code to split the PCA dataset--------------------------------
from sklearn.model_selection import train_test_split
data = pd.read_csv('multi_classification_pca.csv')

# X as features and y as actual labels
X = data.drop(['quality_label'], axis=1)
y = data['quality_label']

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=25, shuffle=True)

# Print dimensions to confirm
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# Save to CSV files
train_set = X_train.copy()
train_set['quality_label'] = y_train
# train_set.to_csv('binary_train_set_pca.csv', index=False)
train_set.to_csv('multi_train_set_pca.csv', index=False)
val_set = X_val.copy()
val_set['quality_label'] = y_val
# val_set.to_csv('binary_validation_set_pca.csv', index=False)
val_set.to_csv('multi_validation_set_pca.csv', index=False)

# print("'binary_train_set_pca.csv' and 'binary_validation_set_pca.csv' saved")
print("'multi_train_set_pca.csv' and 'multi_validation_set_pca.csv' saved")