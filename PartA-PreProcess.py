import pandas as pd

# Load the original dataset
data = pd.read_csv('winequality-red.csv', delimiter=';')

# -----------------------------Problem 1: Binary classification-----------------------------

# Create binary labels
data_binary = data.copy()
data_binary['quality_label'] = data_binary['quality'].apply(lambda q: 1 if q >= 7 else 0)

# Save to CSV
data_binary.to_csv('binary_classification.csv', index=False)
print("Binary classification dataset saved as 'binary_classification.csv'.")

# -----------------------------Problem 2: Multi-class classification-----------------------------


# Create multi-class labels
data_multi = data.copy()
def multiclass_label(q):
    if q <= 5:
        return 0  # Low
    elif q == 6:
        return 1  # Medium
    else:
        return 2  # High

data_multi['quality_label'] = data_multi['quality'].apply(multiclass_label)

# Save to CSV
data_multi.to_csv('multi_classification.csv', index=False)
print("Multi-class classification dataset saved as 'multiclass_classification.csv'.")
