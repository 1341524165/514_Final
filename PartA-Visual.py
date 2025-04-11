import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# data = pd.read_csv('winequality-red.csv')
# ; divider
data = pd.read_csv('winequality-red.csv', delimiter=';')

sns.set(style='whitegrid', palette='muted')

print(data.info())

# Histogram of each feature
data.hist(bins=15, figsize=(15, 10), layout=(4, 3), edgecolor='black')
plt.suptitle('Distribution of Variables', fontsize=16)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Quality Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=data, palette='muted')
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()
