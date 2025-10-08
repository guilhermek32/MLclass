import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Input file path
file_path = 'RTVue_20221110_MLClass.csv'

# Read the CSV file
data_original = pd.read_csv(file_path)

# Display basic information about the dataset
print("Original Dataset:")
print(data_original.head())
print("\n" + "="*100)
print(data_original.info())
print("\n" + "="*100)
print("Descriptive statistics:")
print(data_original.describe())

data = data_original.copy()

# Drop unnecessary columns (Index, pID, Gender, Eye, Age)
data.drop(columns=['Index', 'pID', 'Gender', 'Eye', 'Age'], inplace=True, errors='ignore')

print("\nInitial shape:", data.shape)
# Remove duplicate rows
data.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", data.shape)


# Check for missing values before handling
missing_before = data.isnull().sum()
print("\nMissing values before imputation:\n", missing_before)
print("\nMissing percentage:")
print((missing_before / len(data) * 100).round(2))


# Drop rows with more than 2 missing values
data = data[data.isnull().sum(axis=1) <= 2]

# Handle missing values using median imputation for numeric columns
numeric_columns = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
imputer = SimpleImputer(strategy='median')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])


# Check for missing values after preprocessing
missing_after = data.isnull().sum()
print("\nMissing values after preprocessing:\n", missing_after)

print("\nFinal shape:", data.shape)
print("Final columns:", data.columns.tolist())


# Remove obvious data errors before clustering
data = data[(data['S'] < 300)]  # Remove extreme measurement errors
data = data[(data['IT'] < 300)]  # Similarly for other measurements
data = data[(data['I'] < 300)]
data = data[(data['IN'] < 300)]
data = data[(data['N'] < 300)]
data = data[(data['SN'] < 300)]
data = data[(data['C'] < 160)]
data = data[(data['T'] < 300)]
data = data[(data['ST'] < 300)]

data = data[(data['S'] > 10)]  # Remove non-physical negative values
data = data[(data['IT'] > 10)] # Similarly for other measurements
data = data[(data['I'] > 10)]  # Similarly for other measurements
data = data[(data['IN'] > 10)] # Similarly for other measurements
data = data[(data['N'] > 10)]  # Similarly for other measurements
data = data[(data['SN'] > 10)] # Similarly for other measurements
data = data[(data['C'] > 10)]  # Similarly for other measurements
data = data[(data['T'] > 10)]  # Similarly for other measurements
data = data[(data['ST'] > 10)] # Similarly for other measurements


print("\nShape after removing obvious errors:", data.shape)
print("\n" + "="*100)
print("Descriptive statistics:")
print(data.describe())

# Visualize boxplots for numeric columns
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Save the cleaned data to a new CSV file
data.to_csv('RTVue_20221110_MLClass_cleaned.csv', index=False)