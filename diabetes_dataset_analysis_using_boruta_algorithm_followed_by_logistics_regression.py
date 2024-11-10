import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
data_file_path = os.path.join(script_dir, 'data.csv')
data = pd.read_csv(data_file_path) 
# Display the first few rows of the dataframe
print(data.head())



# Define features and target variable
X = data.drop('target_column', axis=1)  # replace 'target_column' with your target variable
y = data['target_column']  # replace 'target_column' with your target variable

# Initialize Random Forest Classifier for Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Initialize Boruta
boruta_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2,
    random_state=1
)

# Fit Boruta
boruta_selector.fit(X.values, y.values)

# Get selected features
selected_features = X.columns[boruta_selector.support_].tolist()
print("Selected Features: ", selected_features)

# Filter the dataset to keep only selected features
X_selected = X[selected_features]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Regression MSE: {mse_rf}")

# Train Logistic Regression
log_regressor = LogisticRegression(max_iter=1000)
log_regressor.fit(X_train, y_train)

# Make predictions
y_pred_log = log_regressor.predict(X_test)

# Evaluate the model
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {accuracy_log}")

# Train LightGBM Regressor
lgb_regressor = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_regressor.fit(X_train, y_train)

# Make predictions
y_pred_lgb = lgb_regressor.predict(X_test)

# Evaluate the model
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
print(f"LightGBM Regression MSE: {mse_lgb}")