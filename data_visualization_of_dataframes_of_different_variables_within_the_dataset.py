import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Data Visualization of dataframes

# Pie chart
# piechart representation
lat = dataset['Lat']
lon = dataset['Lon']

# generate a piechart to display latitude and longitude of crime rate
plt.figure(figsize=(10,6))
plt.pie(lat.value_counts(), labels=lat.value_counts().index, autopct='%1.1f%%',
        startangle=140)
plt.title("Distribution of Latitude")
plt.axis('equal')
plt.show()

# generate a piechart to display longitude of crime rate
plt.figure(figsize=(10,6))
plt.pie(lon.value_counts(), labels=lon.value_counts().index, autopct='%1.1f%%',
        startangle=140)
plt.title("Distribution of Longitude")
plt.axis('equal')
plt.show()

# time = dataset['Time_Occured']
Time_OCC_counts = Time_OCC.value_counts()
plt.figure(figsize=(10,6))
plt.pie(Time_OCC_counts, labels=Time_OCC_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Time Occured")
plt.axis('equal')
plt.show()


# Find the Status description
status_desc = dataset['Status_Desc']
status_desc_counts = status_desc.value_counts()
plt.figure(figsize=(10,6))
plt.pie(status_desc_counts, labels=status_desc_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of status description")
plt.axis('equal')
plt.show()

# Generate a pie chart based on the time to report days
time_to_report_days = dataset['Time_to_Report_Days']
time_to_report_days_divided = time_to_report_days / 365
time_to_report_days_df = pd.DataFrame({'Time_to_Report_Days_Divided': time_to_report_days_divided})

# Create a bar chart for the divided time to report days
plt.figure(figsize=(10, 6))
time_to_report_days_df['Time_to_Report_Days_Divided'].value_counts().sort_index().plot(kind='bar')
plt.title('Bar Chart of Divided Time to Report Days')
plt.xlabel('Divided Time to Report Days')
plt.ylabel('Frequency')
plt.show()


# Find the severity of crime
severity = dataset['Severity_of_Crime']
# Count occurrences of each severity category
severity_counts = severity.value_counts()

# Generate a pie chart based on the severity of crime
plt.figure(figsize=(10, 6))
plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Severity of Crime')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Sex
sex = dataset['Vict_Sex']
sex_counts = sex.value_counts()

plt.figure(figsize=(10, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=140)  # Corrected autopct format
plt.title('Distribution of Crime based on Sex')
plt.axis('equal')
plt.show()

# Descent
descent = dataset['Vict_Descent']
descent_counts = descent.value_counts()

plt.figure(figsize=(10, 6))
plt.pie(descent_counts, labels=descent_counts.index, autopct='%1.1f%%', startangle=140)  # Corrected labels
plt.title('Distribution based on Descent')
plt.axis('equal')
plt.show()

# Weapon Description
weapon_description = dataset['Weapon_Desc']
weapon_description_counts = weapon_description.value_counts()

plt.figure(figsize=(10, 6))
plt.pie(weapon_description_counts, labels=weapon_description_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution based on Weapon Description')
plt.axis('equal')
plt.show()

# Find the lowest and highest values of the 'Vict_Age' column
min_age = dataset['Vict_Age'].min()
max_age = dataset['Vict_Age'].max()
print(f'Lowest Age: {min_age}, Highest Age: {max_age}')

# Create age groups
bins = [0, 18, 25, 35, 45, 55, 65, 100]  # Define age bins
labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+']  # Define labels for bins
dataset['Age Group'] = pd.cut(dataset['Vict_Age'], bins=bins, labels=labels, right=False)

# Count the number of occurrences in each age group
age_group_counts = dataset['Age Group'].value_counts()

# Generate a pie chart for age groups
plt.figure(figsize=(10, 6))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Age Groups')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Histogram

# Generate a histogram for age distribution
plt.figure(figsize=(10, 6))
plt.hist(dataset['Vict_Age'], bins=bins, edgecolor='black', alpha=0.7)
plt.title('Age Distribution Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')
# plt.figure()
plt.show()

# Barcharts

# Generate a barchart for age group and severity
age_severity_counts = dataset.groupby(['Age Group', 'Severity_of_Crime']).size().unstack()
age_severity_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Age Group and Severity of Crime')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Severity of Crime')
plt.show()

# Create a bar chart showcasing the relationship between Sex and Severity
sex_severity_counts = dataset.groupby(['Vict_Sex', 'Severity_of_Crime']).size().unstack()
sex_severity_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Relationship between Victim Sex and Severity of Crime')
plt.xlabel('Victim Sex')
plt.ylabel('Count')
plt.legend(title='Severity of Crime')
plt.show()

# victim descent and severity
victim_descent_counts = dataset.groupby(['Vict_Descent', 'Severity_of_Crime']).size().unstack()
victim_descent_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Relationship between Victim Descent and Severity of Crime')
plt.xlabel('Victim Descent')
plt.ylabel('Count')
plt.legend(title='Severity of Crime')
plt.show()

# weapon description and severity
weapon_description_counts = dataset.groupby(['Weapon_Desc', 'Severity_of_Crime']).size().unstack()
weapon_description_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Relationship between Weapon Description and Severity of Crime')
plt.xlabel('Weapon Description')
plt.ylabel('Count')
plt.legend(title='Severity of Crime')
plt.show()
