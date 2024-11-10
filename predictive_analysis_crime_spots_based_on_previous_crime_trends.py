import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# predict potential crime spots based on the previous crime trends.
# Convert Severity_of_Crime to numeric
severity_mapping = {
    'Minor Crime': 0,
    'Moderate Crime': 1,
    'Serious Crime': 2,
    'Severe Crime': 3,
    'Grave/Capital Crime': 4
}
dataset['Severity_numeric'] = dataset['Severity_of_Crime'].map(severity_mapping)
# Prepare the feature set
X_crime_location = dataset[['Lat', 'Lon']].copy()
y_crime_severity = dataset['Severity_numeric'].dropna()

# Align X with y
X_crime_location = X_crime_location.loc[y_crime_severity.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_crime_location, y_crime_severity, test_size=0.2, random_state=42)

# Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error for predicting severity of crime: {mse}')
print(f'R^2 Score: {r2}')

kmeans = KMeans(n_clusters=5, random_state=42)  # Choose the number of clusters
dataset['Cluster'] = kmeans.fit_predict(X_crime_location)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Lon'], dataset['Lat'], c=dataset['Cluster'], cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering of Crime Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster')
plt.show()