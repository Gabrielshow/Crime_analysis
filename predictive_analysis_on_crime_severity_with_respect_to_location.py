import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# predict the type of crime that might occur in a specific area based on the time
Time_OCC = dataset['Time_OCC']
Minor_Crime = dataset[dataset['Severity_of_Crime'] == 'Minor Crime']
Moderate_Crime = dataset[dataset['Severity_of_Crime'] == 'Moderate Crime']
Serious_Crime = dataset[dataset['Severity_of_Crime'] == 'Serious Crime']
Severe_Crime = dataset[dataset['Severity_of_Crime'] == 'Severe Crime']
Grave_Capital_Crime = dataset[dataset['Severity_of_Crime'] == 'Grave/Capital Crime']

# crime dataset with respect to age
Minor_Crime_age = Minor_Crime['Vict_Age']
Moderate_Crime_age = Moderate_Crime['Vict_Age']
Serious_Crime_age = Serious_Crime['Vict_Age']
Severe_Crime_age = Severe_Crime['Vict_Age']
Grave_Capital_Crime_age = Grave_Capital_Crime['Vict_Age']

# Crime dataset after numeric conversion
Minor_Crime['Crime_numeric'] = 0
Moderate_Crime['Criime_numeric'] = 1
Serious_Crime['Crime_numeric'] = 2
Severe_Crime['Crime_numeric'] = 3
Grave_Capital_Crime['Crime_numeric'] = 4

X_crime = pd.concat([Minor_Crime[['Crime_numeric']],
               Moderate_Crime[['Crime_numeric']],
               Serious_Crime[['Crime_numeric']],
               Severe_Crime[['Crime_numeric']],
               Grave_Capital_Crime[['Crime_numeric']]], axis=1)

y_crime = dataset['Vict_Age'].dropna()
X_crime = X_crime.loc[y_crime.index]
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error for Victim sex and Victim age: {mse}')
print(f'R^2 Score: {r2}')