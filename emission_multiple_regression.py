# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:39:14 2019

@author: Tisi
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

'''Data Preposessing '''

# Importing the dataset
dataset = pd.read_csv('emission_clean.csv')

# Encoding categorical X features
X = dataset.iloc[:, 3:]

for col in X.columns:  #check X dataframe 
    print(col)
    
Xdata = pd.get_dummies(X, prefix = ['state', 'iso_name', 'nerc_region', 'acid_prog' ], drop_first=True) 

#set matrix of variables
X = Xdata.values
y = dataset.iloc[:, 1].values

#visualize y variable
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(dataset['carbon'])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_test = sc_y.fit_transform(y_test.reshape(-1, 1))

'''Building Models '''
#Multiple Linear Regression 
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
coeff_df = pd.DataFrame(regressor.coef_.flatten(), Xdata.columns)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Compare actual and predicted values
compare = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

#visualize the comparison
df1 = compare.head(1000)
df1.plot(kind='line',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Multiple Linear Regression (Test set)')
plt.show()

#Evaluate the performance of the algorithm
import statistics
print (statistics.mean(y_test.flatten()))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sns.residplot(X_test[:,1], y_test)
sns.residplot(y_pred, y_test)













