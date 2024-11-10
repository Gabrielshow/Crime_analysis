import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# predict demographic attributes of a victim (age, gender and descent) based on time type, location and time of occurrence.
# Convert categorical variables to numeric
sex_mapping = {'M': 1, 'F': 2, 'X': 0}  # 0 for Other
descent_mapping = {'H': 1, 'X': 2, 'O': 3, 'B': 4, 'W': 5}

dataset['Sex_numeric'] = dataset['Vict_Sex'].map(sex_mapping)
dataset['Descent_numeric'] = dataset['Vict_Descent'].map(descent_mapping)

# Prepare features: Lat, Lon, Time_OCC, and Time_Occured (you may need to preprocess Time_OCC)
# Convert Time_OCC to a suitable numeric format (e.g., hour of the day)
dataset['Time_OCC_hour'] = pd.to_datetime(dataset['Time_OCC'], format='%H:%M:%S').dt.hour
# Create feature set
X = dataset[['Lat', 'Lon', 'Time_OCC_hour']]

# Target variables
y_age = dataset['Vict_Age'].dropna()
y_gender = dataset['Sex_numeric'].dropna()
y_descent = dataset['Descent_numeric'].dropna()

# Ensure alignment of features with target variables
X_age = X.loc[y_age.index]
X_gender = X.loc[y_gender.index]
X_descent = X.loc[y_descent.index]

# Predicting Age
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_age, y_age, test_size=0.2, random_state=42)
model_age = LinearRegression()
model_age.fit(X_train_age, y_train_age)
y_pred_age = model_age.predict(X_test_age)

mse_age = mean_squared_error(y_test_age, y_pred_age)
r2_age = r2_score(y_test_age, y_pred_age)
print(f'Mean Squared Error for Age: {mse_age}')
print(f'R^2 Score for Age: {r2_age}')

# Predicting Gender
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)
model_gender = LinearRegression()
model_gender.fit(X_train_gender, y_train_gender)
y_pred_gender = model_gender.predict(X_test_gender)

mse_gender = mean_squared_error(y_test_gender, y_pred_gender)
r2_gender = r2_score(y_test_gender, y_pred_gender)
print(f'Mean Squared Error for Gender: {mse_gender}')
print(f'R^2 Score for Gender: {r2_gender}')

# Predicting Descent
X_train_descent, X_test_descent, y_train_descent, y_test_descent = train_test_split(X_descent, y_descent, test_size=0.2, random_state=42)
model_descent = LinearRegression()
model_descent.fit(X_train_descent, y_train_descent)
y_pred_descent = model_descent.predict(X_test_descent)

mse_descent = mean_squared_error(y_test_descent, y_pred_descent)
r2_descent = r2_score(y_test_descent, y_pred_descent)
print(f'Mean Squared Error for Descent: {mse_descent}')
print(f'R^2 Score for Descent: {r2_descent}')

# Using K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Choose the number of clusters
dataset['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=( 10, 6))
plt.scatter(dataset['Lon'], dataset['Lat'], c=dataset['Cluster'], cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering of Victim Demographics')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster')
plt.show()



