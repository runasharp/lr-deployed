import pandas as pd
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import pickle

# Load data set
data = pd.read_csv('dataset.csv')

# Name the columns
data.columns = ['complexAge', 'totalRooms', 'totalBedrooms',
                'complexInhabitants', 'apartmentsNr', 'medianComplexValue']

# Independent variables are stored in X
X = data[['complexAge', 'totalRooms', 'totalBedrooms',
          'complexInhabitants', 'apartmentsNr']].values
# # Dependent variable (price) is stored in Y
Y = data['medianComplexValue'].values  # values converts it into a numpy array

# Split the data into training/testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

regr = LinearRegression(normalize=True)  # creates object for the class
clf = regr.fit(X_train, Y_train)  # performs linear regression
Y_pred = regr.predict(X_test)  # makes predict
print('This is it', Y_pred)

# The coefficients
print('Slope coefficients: \n', regr.coef_)

print('Intercept: \n', regr.intercept_)
# The mean squared error
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f\n'
      % r2_score(Y_test, Y_pred))

print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, Y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, Y_pred), 2))


# x1 = int(input("Enter complexAge\n"))
# x2 = int(input("Enter totalRooms\n"))
# x3 = int(input("Enter totalBedrooms\n"))
# x4 = int(input("Enter complexInhabitants\n"))
# x5 = int(input("Enter apartmentsNr\n"))
# inputs = [[x1, x2, x3, x4, x5]]

# inputs = [[52, 2491, 474, 1098, 468]]  # expected 213500, got around 269000
# prediction = regr.predict(inputs).flatten()

# print("Predicted value", prediction)


with open('regr.pkl', 'wb') as fid:
    pickle.dump(regr, fid, 2)