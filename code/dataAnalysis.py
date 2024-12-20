import pandas as pd

file_path = './data/fraud_detection_dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Display basic information about the dataset
print("\nDataset Info:")
print(data.info())

# Check for missing values
print("\nMissing Values Count:")
print(data.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of FICO scores
plt.figure(figsize=(8, 5))
sns.histplot(data['fico_score'], kde=True, bins=30, color='blue')
plt.title('Distribution of FICO Scores')
plt.xlabel('FICO Score')
plt.ylabel('Frequency')
plt.show()

# Debt-to-Income Ratio vs. Charge-off Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='charge_off_status', y='debt_to_income_ratio', data=data, hue='charge_off_status', palette='Set2', legend=False)
plt.title('Debt-to-Income Ratio by Charge-off Status')
plt.xlabel('Charge-off Status')
plt.ylabel('Debt-to-Income Ratio')
plt.show()

# Delinquency Status by Charge-off Status
plt.figure(figsize=(8, 5))
sns.boxplot(x='charge_off_status', y='delinquency_status', data=data, hue='charge_off_status', palette='Set3', legend=False)
plt.title('Delinquency Status by Charge-off Status')
plt.xlabel('Charge-off Status')
plt.ylabel('Delinquency Status')
plt.show()

# Correlation heatmap (numerical features)
plt.figure(figsize=(12, 8))
corr_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

