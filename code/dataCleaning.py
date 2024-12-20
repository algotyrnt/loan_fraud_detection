import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('./data/fraud_detection_dataset.csv')

# Handle missing values for numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer_mean = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_mean.fit_transform(data[numerical_cols])

# Standardize the date format for 'account_open_date'
if 'account_open_date' in data.columns:
    data['account_open_date'] = pd.to_datetime(data['account_open_date'], errors='coerce')
    data['account_open_days'] = (data['account_open_date'] - data['account_open_date'].min()).dt.days
    data = data.drop(columns=['account_open_date'])

# Encode categorical variables using one-hot encoding
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop remaining rows with missing values
data = data.dropna()

# Save cleaned data
data.to_csv('./data/cleaned_fraud_detection_data.csv', index=False)

print("Data cleaning and preprocessing completed. Cleaned data saved as 'cleaned_fraud_detection_data.csv'.")
