from time import time_ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import lightgbm as lgb

# Reference the Excel file directly
FILE_NAME = "Cleaned Crime Dataset Bukkies.xlsx"  # Ensure the correct file extension

# Load the dataset
dataset = pd.read_excel(FILE_NAME)  # Load the dataset from the current directory

# Check the first few rows of the dataset to understand its structure
print(dataset.head())

# Convert categorical variables to numeric using .loc
dataset.loc[dataset['Vict_Sex'] == 'M', 'Sex_numeric'] = 1
dataset.loc[dataset['Vict_Sex'] == 'F', 'Sex_numeric'] = 2
dataset.loc[dataset['Vict_Sex'] == 'X', 'Sex_numeric'] = 0

dataset.loc[dataset['Vict_Descent'] == 'H', 'Descent_numeric'] = 1
dataset.loc[dataset['Vict_Descent'] == 'X', 'Descent_numeric'] = 2
dataset.loc[dataset['Vict_Descent'] == 'O', 'Descent_numeric'] = 3
dataset.loc[dataset['Vict_Descent'] == 'B', 'Descent_numeric'] = 4
dataset.loc[dataset['Vict_Descent'] == 'W', 'Descent_numeric'] = 5

# Drop rows with NaN values in the relevant columns
dataset.dropna(subset=['Vict_Age', 'Sex_numeric', 'Descent_numeric'], inplace=True)

# Prepare features and target variable for regression
X = dataset[['Sex_numeric', 'Descent_numeric']]
y = dataset['Vict_Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest - Mean Squared Error: {mse_rf}')
print(f'Random Forest - R^2 Score: {r2_rf}')

# Prepare features and target variable for classification
dataset['Crime_numeric'] = dataset['Severity_of_Crime'].map({
    'Minor Crime': 0,
    'Moderate Crime': 1,
    'Serious Crime': 2,
    'Severe Crime': 3,
    'Grave/Capital Crime': 4
})

# Drop rows with NaN values in the classification target
dataset.dropna(subset=['Crime_numeric'], inplace=True)

X_class = dataset[['Sex_numeric', 'Descent_numeric']]
y_class = dataset['Crime_numeric']

# Split the data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_class, y_train_class)
y_pred_gb = gb_model.predict(X_test_class)

accuracy_gb = accuracy_score(y_test_class, y_pred_gb)
print(f'Gradient Boosting - Accuracy: {accuracy_gb}')

# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train_class, y_train_class)
y_pred_lgb = lgb_model.predict(X_test_class)

accuracy_lgb = accuracy_score(y_test_class, y_pred_lgb)
print(f'LightGBM - Accuracy: {accuracy_lgb}')

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_class, y_train_class)
y_pred_log = log_model.predict(X_test_class)

accuracy_log = accuracy_score(y_test_class, y_pred_log)
print(f'Logistic Regression - Accuracy: {accuracy_log}')