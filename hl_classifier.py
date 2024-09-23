"""
Binary Classifier of Hearing Impairment with Gradient Boosting

Script generates synthetic audiology data and uses a Gradient Boosting Classifier to predict hearing impairment.
Key features include SelectKBest for feature selection, hyperparameter tuning using GridSearchCV, cross-validation,
and evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
"""

################################################################################################################################################
# Import libraries
################################################################################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

################################################################################################################################################
# Generate data
################################################################################################################################################
# Input data
np.random.seed(42)
n_samples = 5000

# Features: Age, Pure-tone average (PTA) in dB, Speech discrimination score (SDS), Tympanometry score
X = pd.DataFrame({
    'Age': np.random.randint(20, 100, n_samples),
    'PTA': np.random.normal(30, 12, n_samples),  # Pure-tone average in dB
    'Speech': np.random.normal(80, 5, n_samples),  # Speech recognition score in %
    'Tympanometry': np.random.normal(1.5, 0.5, n_samples),  # Tympanometry score
    'Noise_Exposure_Years': np.random.randint(0, 40, n_samples),  # Years of noise exposure
    'Family_History_HL': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # Binary: 30% have family history of HL
    'Occupation': np.random.choice(['Office', 'Construction', 'Factory', 'Music'], n_samples, p=[0.5, 0.2, 0.2, 0.1]),  # Categorical
})

# Convert 'Occupation' to dummy variables for model use
X = pd.get_dummies(X, columns=['Occupation'], drop_first=True)

# Binary target: 1 = Hearing impairment, 0 = No hearing impairment
# Hearing impairment is influenced by a combination of PTA, Age, and Speech, but with some randomness
noise = np.random.rand(n_samples)
y = (((X['PTA'] > 40) & (X['Age'] > 65)) | (X['Speech'] < 70) | (noise > 0.85)).astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SelectKBest feature selection
k_best = SelectKBest(score_func=f_classif, k=4)  # Select top 3 features
X_train_k_best = k_best.fit_transform(X_train_scaled, y_train)
X_test_k_best = k_best.transform(X_test_scaled)

################################################################################################################################################
# Build model
################################################################################################################################################
# Define the model
gb = GradientBoostingClassifier(random_state=42)

# Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, scoring='accuracy', cv=5, verbose=0)

# Train the model using GridSearchCV
grid_search.fit(X_train_k_best, y_train)

################################################################################################################################################
# Make predictions with cross-validation
################################################################################################################################################
# Make predictions on the test set
y_pred = grid_search.predict(X_test_k_best)
y_pred_proba = grid_search.predict_proba(X_test_k_best)[:, 1]  # For ROC-AUC

# Output the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Cross-validation using the best model from GridSearchCV
best_model = grid_search.best_estimator_
cross_val_scores = cross_val_score(best_model, X_train_k_best, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Accuracy: {cross_val_scores.mean() * 100:.2f}%")

################################################################################################################################################
# Evaluate model
################################################################################################################################################
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Precision, Recall, and F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.2f}")

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

################################################################################################################################################
# Plot ROC curve
################################################################################################################################################
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.rcParams['font.family'] = 'Calibri'
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontweight='bold',fontsize=13)
plt.ylabel('True Positive Rate', fontweight='bold',fontsize=13)
plt.title('Receiver Operating Characteristic Curve', fontweight='bold',fontsize=16)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

################################################################################################################################################
# Plot Feature Importance
################################################################################################################################################
importances = best_model.feature_importances_
feature_names = X_train.columns[k_best.get_support()]  # Get the names of the selected features
plt.figure()
plt.rcParams['font.family'] = 'Calibri'
plt.barh(feature_names, importances, color='teal')
plt.xlabel('Feature Importance', fontweight='bold',fontsize=13)
plt.ylabel('Feature', fontweight='bold',fontsize=13)
plt.title('Feature Importance', fontweight='bold',fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
