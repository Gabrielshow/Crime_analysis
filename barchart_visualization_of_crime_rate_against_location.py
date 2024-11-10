import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# crime rate and location

# Convert Severity_of_Crime to numeric if not already done
severity_mapping = {
    'Minor Crime': 0,
    'Moderate Crime': 1,
    'Serious Crime': 2,
    'Severe Crime': 3,
    'Grave/Capital Crime': 4
}
dataset['Severity_numeric'] = dataset['Severity_of_Crime'].map(severity_mapping)

# Group by Latitude and Longitude and count occurrences of each severity
location_severity_counts = dataset.groupby(['Lat', 'Lon', 'Severity_numeric']).size().unstack(fill_value=0)

# Reset index to make it easier to plot
location_severity_counts.reset_index(inplace=True)

# Plotting the bar chart
location_severity_counts.set_index(['Lat', 'Lon']).plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Crime Severity by Location (Latitude and Longitude)')
plt.xlabel('Location (Latitude, Longitude)')
plt.ylabel('Count of Crimes')
plt.legend(title='Severity of Crime', labels=['Minor Crime', 'Moderate Crime', 'Serious Crime', 'Severe Crime', 'Grave/Capital Crime'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()