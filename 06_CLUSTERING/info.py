import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Input .xlsx file path
file_path = 'barrettII_eyes_clustering.xlsx'

# Read the Excel file
data_original = pd.read_excel(file_path)

# Display basic information about the dataset
print(data_original.head())
print(data_original.info())

data = data_original.copy()
data = data.drop(columns=['ID', 'Correto'])

# Change extension to .csv
data.to_csv('barrettII_eyes_clustering.csv', index=False)

# Display one boxplot only
plt.figure(figsize=(8, 6))
sns.boxplot(data=data)

# Display all boxplots
plt.figure(figsize=(12, 8))


# Count values of AL column where AL > 25 and AL < 22
count_above_25 = (data['AL'] > 25).sum()
count_below_22 = (data['AL'] < 22).sum()
print(f"Count of AL > 25: {count_above_25}")
print(f"Count of AL < 22: {count_below_22}")

# Count values of ACD column where ACD > 3.6 and ACD < 2.5
count_acd_above_36 = (data['ACD'] > 3.6).sum()
count_acd_below_25 = (data['ACD'] < 2.5).sum()
print(f"Count of ACD > 3.6: {count_acd_above_36}")
print(f"Count of ACD < 2.5: {count_acd_below_25}")

# Count values of WTW column where WTW > 12.8 and WTW < 10.8
count_wtw_above_128 = (data['WTW'] > 12.8).sum()
count_wtw_below_108 = (data['WTW'] < 10.8).sum()
print(f"Count of WTW > 12.8: {count_wtw_above_128}")
print(f"Count of WTW < 10.8: {count_wtw_below_108}")

# Count values of K1 column where K1 > 48 and K1 < 36
count_k1_above_48 = (data['K1'] > 48).sum()
count_k1_below_36 = (data['K1'] < 36).sum()
print(f"Count of K1 > 48: {count_k1_above_48}")
print(f"Count of K1 < 36: {count_k1_below_36}")

# Count values of K2 column where K2 > 48 and K2 < 36
count_k2_above_48 = (data['K2'] > 48).sum()
count_k2_below_36 = (data['K2'] < 36).sum()
print(f"Count of K2 > 48: {count_k2_above_48}")
print(f"Count of K2 < 36: {count_k2_below_36}")

# Print the max and min values
max_values = data.max()
min_values = data.min()
print("Max values:")
print(max_values)
print("\nMin values:")
print(min_values)

# Print the mean and median values
mean_values = data.mean()
median_values = data.median()
print("\nMean values:")
print(mean_values)
print("\nMedian values:")
print(median_values)
