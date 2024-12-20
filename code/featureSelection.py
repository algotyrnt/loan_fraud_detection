import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/cleaned_fraud_detection_data.csv')

# Define target variable and features
target = 'charge_off_status'
features = data.drop(columns=[target])
labels = data[target]

# Feature selection
selector = SelectKBest(score_func=f_classif, k=50)
selected_features = selector.fit_transform(features, labels)

# Get the selected feature names
selected_feature_names = features.columns[selector.get_support()]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    selected_features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save feature selection data
selected_features_df = pd.DataFrame(selected_features, columns=selected_feature_names)
X_train_df = pd.DataFrame(X_train, columns=selected_feature_names)
X_test_df = pd.DataFrame(X_test, columns=selected_feature_names)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

X_train_df.to_csv('./data/X_train.csv', index=False)
X_test_df.to_csv('./data/X_test.csv', index=False)
y_train_df.to_csv('./data/y_train.csv', index=False)
y_test_df.to_csv('./data/y_test.csv', index=False)

print("Feature selection and data splitting completed. Files saved as 'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv'.")
