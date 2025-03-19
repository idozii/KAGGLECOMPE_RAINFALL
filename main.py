import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Feature correlation with target
plt.figure(figsize=(12, 10))
corr = train_data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Prepare data for modeling
X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']
test_ids = test_data['id']
X_test = test_data.drop('id', axis=1)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_val_preds = rf_model.predict_proba(X_val)[:, 1]
rf_val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
rf_val_auc = roc_auc_score(y_val, rf_val_preds)

# Print validation results
print(f"\nRandom Forest - Validation Accuracy: {rf_val_accuracy:.4f}, AUC: {rf_val_auc:.4f}")

# Make predictions on test data
test_preds_proba = rf_model.predict_proba(X_test)[:, 1]
test_preds_binary = rf_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall_probability': test_preds_proba
})

submission.to_csv('rainfall_predictions.csv', index=False)
print("\nPredictions saved to 'rainfall_predictions.csv'")
