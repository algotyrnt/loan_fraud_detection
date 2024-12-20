import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import joblib

# Load the training and testing datasets
X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv').values.flatten()
y_test = pd.read_csv('./data/y_test.csv').values.flatten()

# Initialize and train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display model evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)
print("Confusion Matrix:\n", conf_matrix)

# SHAP: Ensure feature consistency between training and test data
# Align columns of X_test to match the features used in training
X_test = X_test[X_train.columns]

# Explainability: SHAP values (for binary classification, using shap_values[1])
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Check shapes of shap_values and X_test
print(f"Shape of SHAP values (positive class): {shap_values[1].shape}")
print(f"Shape of X_test: {X_test.shape}")

# Ensure the correct SHAP values for the positive class (shap_values[1])
# Make sure we're plotting the correct SHAP values matrix
shap.summary_plot(shap_values[1], X_test, feature_names=X_test.columns)

# Save the model for future use
joblib.dump(model, './model/fraud_detection_model.pkl')

print("Model training completed. Model saved as 'fraud_detection_model.pkl'.")
