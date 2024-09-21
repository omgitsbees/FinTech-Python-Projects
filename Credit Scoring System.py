import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Simulating feature columns
income = np.random.normal(50000, 15000, n_samples)
age = np.random.normal(40, 10, n_samples)
loan_amount = np.random.normal(15000, 5000, n_samples)
credit_history = np.random.randint(0, 10, n_samples)

# Default (target) column based on some linear combination of the features
coefficients = np.array([0.00003, -0.02, 0.0001, -0.1])  # Coefficients for the features
intercept = -4

linear_combination = (
    coefficients[0] * income +
    coefficients[1] * age +
    coefficients[2] * loan_amount +
    coefficients[3] * credit_history +
    intercept
)

# Convert linear combination to probability using sigmoid function
probability = 1 / (1 + np.exp(-linear_combination))

# Generate binary outcomes (default: 1, no default: 0)
default = np.random.binomial(1, probability)

# Create DataFrame
data = pd.DataFrame({
    'income': income,
    'age': age,
    'loan_amount': loan_amount,
    'credit_history': credit_history,
    'default': default
})

# View first few rows of the data
print(data.head())

# Split into features (X) and target (y)
X = data.drop('default', axis=1)
y = data['default']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Enhanced Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = [f'{value:0.0f}' for value in conf_matrix.flatten()]
group_percentages = [f'{value:.2%}' for value in conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f'{name}\n{count}\n{percentage}' for name, count, percentage in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title('Confusion Matrix with Labels')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print Precision, Recall, F1 Score
print('Classification Report:\n', classification_report(y_test, y_pred))

# Alternatively, calculate precision, recall, and f1 score manually
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# ROC-AUC Score and ROC Curve
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'ROC-AUC Score: {roc_auc}')
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()