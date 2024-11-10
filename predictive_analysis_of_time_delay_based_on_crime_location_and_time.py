import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

severity_mapping = {
    'Minor Crime': 0,
    'Moderate Crime': 1,
    'Serious Crime': 2,
    'Severe Crime': 3,
    'Grave/Capital Crime': 4
}
dataset['Severity_numeric'] = dataset['Severity_of_Crime'].map(severity_mapping)

# Prepare the feature set
X_reporting = dataset[['Severity_numeric', 'Lat', 'Lon']].copy()

# Assuming 'Time_OCC' is in a suitable format, if not, convert it
dataset['Time_OCC_hour'] = pd.to_datetime(dataset['Time_OCC'], format='%H:%M:%S').dt.hour
X_reporting['Time_OCC_hour'] = dataset['Time_OCC_hour']

# Target variable
y_reporting = dataset['Time_to_Report_Days'].dropna()

# Align X with y
X_reporting = X_reporting.loc[y_reporting.index]
# Split the data into training and testing sets
X_train_reporting, X_test_reporting, y_train_reporting, y_test_reporting = train_test_split(X_reporting, y_reporting, test_size=0.2, random_state=42)

# Fit a Linear Regression model
model_reporting = LinearRegression()
model_reporting.fit(X_train_reporting, y_train_reporting)

# Predictions
y_pred_reporting = model_reporting.predict(X_test_reporting)

# Evaluate the model
mse_reporting = mean_squared_error(y_test_reporting, y_pred_reporting)
r2_reporting = r2_score(y_test_reporting, y_pred_reporting)
print(f'Mean Squared Error for predicting delay in reporting: {mse_reporting}')
print(f'R^2 Score: {r2_reporting}')

# Using K-Means clustering
kmeans_reporting = KMeans(n_clusters=5, random_state=42)  # Choose the number of clusters
dataset['Cluster'] = kmeans_reporting.fit_predict(X_reporting)
# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Lon'], dataset['Lat'], c=dataset['Cluster'], cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering of Reporting Delays')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster')
plt.show()