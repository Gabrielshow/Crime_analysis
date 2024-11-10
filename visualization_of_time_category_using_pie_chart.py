import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Convert 'Time_OCC' to datetime format if it's not already
dataset['Time_OCC'] = pd.to_datetime(dataset['Time_OCC'], format='%H:%M:%S', errors='coerce')

# Create time categories based on the specified ranges
time_bins = [
    pd.Timestamp('06:00:00'),  # 6:00 AM
    pd.Timestamp('12:00:00'),  # 12:00 PM
    pd.Timestamp('17:00:00'),  # 5:00 PM
    pd.Timestamp('21:00:00'),  # 9:00 PM
    pd.Timestamp('05:59:59')   # 5:59 AM (next day)
]

time_labels = ['6-11:59', '12-16:59', '17-20:59', '21-5:59']

# Create a new column for the time categories
dataset['Time_Category'] = pd.cut(dataset['Time_OCC'].dt.hour,
                                    bins=[6, 12, 17, 21, 24, 6],
                                    labels=time_labels,
                                    right=False)

# Count occurrences of each time category
time_category_counts = dataset['Time_Category'].value_counts()

# Generate a pie chart for the time categories
plt.figure(figsize=(10, 6))
plt.pie(time_category_counts, labels=time_category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Crimes by Time of Day")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()