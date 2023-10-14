import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix
import joblib

# Data Processing
data = pd.read_csv("C:\\Users\\user\\OneDrive\\Documents\\AER850\\Project 1 Data.csv")
print(data.columns, "\n")

# Data Visualization
step_number = 1
step_data = data[data['Step'] == step_number]
plt.figure(figsize=(12, 6))
plt.hist(step_data['X'], bins=20, alpha=0.5, label='X Coordinate')
plt.hist(step_data['Y'], bins=20, alpha=0.5, label='Y Coordinate')
plt.hist(step_data['Z'], bins=20, alpha=0.5, label='Z Coordinate')
plt.xlabel('Coordinate Value')
plt.ylabel('Frequency')
plt.title(f'Coordinate Distribution for Step {step_number}')
plt.legend()
plt.show()

# Correlation Analysis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Model Development and Training
X = data[['X', 'Y', 'Z']]
y = data['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier with specified hyperparameters
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)

# Model Performance Analysis
y_predict = rf_model.predict(X_test)
f1 = f1_score(y_test, y_predict, average='weighted')
precision = precision_score(y_test, y_predict, average='weighted')
accuracy = accuracy_score(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)

print('F1 Score:', f1)
print('Precision:', precision)
print('Accuracy:', accuracy)
print('\nConfusion Matrix:')
print(conf_matrix)

# Model Evaluation and Saving
best_model = rf_model
joblib.dump(best_model, 'best_model.joblib')

# Prediction
loaded_model = joblib.load('best_model.joblib')
new_coordinates = np.array([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]])
predictions = loaded_model.predict(new_coordinates)
print('\nPredictions for new coordinates:', predictions)
