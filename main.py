import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Check for missing values
print("Missing values in train data:")
print(train_data.isnull().sum())
print("\nMissing values in test data:")
print(test_data.isnull().sum())

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

# Handle missing values using an imputer
print("\nImputing missing values...")
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Verify no more missing values
print(f"Missing values in training features after imputation: {X.isnull().sum().sum()}")
print(f"Missing values in test features after imputation: {X_test.isnull().sum().sum()}")

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training multiple models...")

# Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_val_preds = rf_model.predict_proba(X_val)[:, 1]
rf_val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
rf_val_auc = roc_auc_score(y_val, rf_val_preds)
print(f"Random Forest - Validation Accuracy: {rf_val_accuracy:.4f}, AUC: {rf_val_auc:.4f}")

# Model 2: Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_val_preds = gb_model.predict_proba(X_val)[:, 1]
gb_val_accuracy = accuracy_score(y_val, gb_model.predict(X_val))
gb_val_auc = roc_auc_score(y_val, gb_val_preds)
print(f"Gradient Boosting - Validation Accuracy: {gb_val_accuracy:.4f}, AUC: {gb_val_auc:.4f}")

# Model 3: XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_val_preds = xgb_model.predict_proba(X_val)[:, 1]
xgb_val_accuracy = accuracy_score(y_val, xgb_model.predict(X_val))
xgb_val_auc = roc_auc_score(y_val, xgb_val_preds)
print(f"XGBoost - Validation Accuracy: {xgb_val_accuracy:.4f}, AUC: {xgb_val_auc:.4f}")

# Model 4: LightGBM
lgb_model = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_val_preds = lgb_model.predict_proba(X_val)[:, 1]
lgb_val_accuracy = accuracy_score(y_val, lgb_model.predict(X_val))
lgb_val_auc = roc_auc_score(y_val, lgb_val_preds)
print(f"LightGBM - Validation Accuracy: {lgb_val_accuracy:.4f}, AUC: {lgb_val_auc:.4f}")

# Probability calibration
print("\nPerforming probability calibration...")

# Calibrate Random Forest
calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
calibrated_rf.fit(X_train, y_train)
cal_rf_val_preds = calibrated_rf.predict_proba(X_val)[:, 1]
cal_rf_val_auc = roc_auc_score(y_val, cal_rf_val_preds)
print(f"Calibrated Random Forest - Validation AUC: {cal_rf_val_auc:.4f}")

# Calibrate XGBoost
calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
calibrated_xgb.fit(X_train, y_train)
cal_xgb_val_preds = calibrated_xgb.predict_proba(X_val)[:, 1]
cal_xgb_val_auc = roc_auc_score(y_val, cal_xgb_val_preds)
print(f"Calibrated XGBoost - Validation AUC: {cal_xgb_val_auc:.4f}")

# Plot calibration curves
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

# Original models
prob_true, prob_pred = calibration_curve(y_val, rf_val_preds, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Random Forest (uncalibrated)')

prob_true, prob_pred = calibration_curve(y_val, xgb_val_preds, n_bins=10)
plt.plot(prob_pred, prob_true, marker='s', label='XGBoost (uncalibrated)')

# Calibrated models
prob_true, prob_pred = calibration_curve(y_val, cal_rf_val_preds, n_bins=10)
plt.plot(prob_pred, prob_true, marker='v', label='Random Forest (calibrated)')

prob_true, prob_pred = calibration_curve(y_val, cal_xgb_val_preds, n_bins=10)
plt.plot(prob_pred, prob_true, marker='^', label='XGBoost (calibrated)')

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curves')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('calibration_curves.png')
plt.close()

# Create ensemble predictions (weighted average)
print("\nCreating ensemble of calibrated models...")

# Find the best weights based on validation AUC scores
weights = [cal_rf_val_auc, gb_val_auc, cal_xgb_val_auc, lgb_val_auc]
weights = np.array(weights) / sum(weights)  # Normalize weights to sum to 1
print(f"Model weights: {weights}")

# Create weighted ensemble predictions for validation set
ensemble_val_preds = (
    weights[0] * cal_rf_val_preds + 
    weights[1] * gb_val_preds + 
    weights[2] * cal_xgb_val_preds + 
    weights[3] * lgb_val_preds
)
ensemble_val_auc = roc_auc_score(y_val, ensemble_val_preds)
print(f"Ensemble - Validation AUC: {ensemble_val_auc:.4f}")

# Make predictions on test data with each model
rf_test_preds = calibrated_rf.predict_proba(X_test)[:, 1]
gb_test_preds = gb_model.predict_proba(X_test)[:, 1]
xgb_test_preds = calibrated_xgb.predict_proba(X_test)[:, 1]
lgb_test_preds = lgb_model.predict_proba(X_test)[:, 1]

# Create weighted ensemble predictions for test set
test_preds_proba = (
    weights[0] * rf_test_preds + 
    weights[1] * gb_test_preds + 
    weights[2] * xgb_test_preds + 
    weights[3] * lgb_test_preds
)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_preds_proba
})

submission.to_csv('submission.csv', index=False)