# - - - - Titanic Survivor Prediction - - - -
# • Kaggle Practice Competition
# • Attempt using K Means Algrithm

# Importing needed packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Loading train and test data from csv
titanicTrain = pd.read_csv('Data/train.csv')
titanicTest = pd.read_csv('Data/test.csv')

# - - - Data Functions - - -

# Function to convert binary column data into numbers
def numerizeBinary(object):
  # One line if-else statement
  # print(object)
  return 0 if object == 'male' else 1

# Function to OneHot encode columns with >2 different values
# • column -> Column to encode
# • prefix -> Prefix for encoded column so they can be recognized later
def oneHotize(data, column, prefix, merge=False, dropNaN=None):
  # Defining OneHot encoder
  encoder = preprocessing.OneHotEncoder()
  # Reshaping array into 1D vector for encoder
  columnData = np.array(data[column]).reshape(-1, 1)
  # Applying OneHot encoder to data
  encodedData = encoder.fit_transform(columnData).toarray()
  # Setting up new column names
  columns = [(prefix + column) for column in encoder.categories_[0]]
  # Creating dataframe with encoded data
  encodedFrame = pd.DataFrame(encodedData, columns=columns)
  # Returning data
  return mergeOneHot(data, column, encodedFrame, prefix, dropNaN) if merge else encodedFrame

def mergeOneHot(data, original, encoded, prefix, dropNaN=None):
  merged = pd.concat([data, encoded], axis=1).drop(original, axis=1)
  return merged.drop(prefix + dropNaN, axis=1) if dropNaN else merged


# - - - Cleaning and Optimizing Data - - -

# Columns:
# • 'PassengerId' - Standard numeric ID
# • 'Survived' - 1: yes, 0: No
# • 'Pclass' - 1: First Class, 2: Second Class, 3: Third Class -> First Class is best
# • 'Name' - Person's name
# • 'Sex' - Person's sex
# • 'Age' - Person's age in years
# • 'SibSp' - Number of person's Siblings & Spouses on board
# • 'Parch' - Number of person's Parents & Children on board
# • 'Ticket' - Ticket Number
# • 'Fare' - Fare paid for ticket
# • 'Cabin' - Cabin Number
# • 'Embarked' - Port from which departed, C = Cherbourg, Q = Queenstown, S = Southampton (X=NaN)

# Experimenting which columns should and can be dropped
titanicTrain = titanicTrain.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1)

# Splitting data into features and labels for validation
X = titanicTrain.drop('Survived', axis=1)
y = titanicTrain['Survived']

# Filling holes in data
# Replacing NaN Cabin number with string 'None'
X['Cabin'].fillna('None', inplace=True)
# Replacing NaN Port where Embarked with string 'X'
X['Embarked'].fillna('X', inplace=True)
# Replacing NaN Age with float 0
X['Age'].fillna(0, inplace=True)

# - Cleaning Columns -

## • Column: 'Sex'
# Applying 'numerizeBinary' function to each row in 'Sex' column
X['Sex'] = list(map(numerizeBinary, titanicTrain['Sex']))


## • Column: 'Embarked'
# OneHot encoding 'Embarked'
# Dropping 'X' column containing NaNs
prefix = 'Emb_'
X = oneHotize(X, 'Embarked', prefix, merge=True, dropNaN='X')


## • Column: 'Cabin'
# OneHot encoding 'Cabin'
# Dropping 'None' column containing NaNs
prefix = 'Cab_'
X = oneHotize(X, 'Cabin', prefix, merge=True, dropNaN='None')


# Scaling data to optimize performance
# X = preprocessing.scale(X)

# Defining classifier
clf = KMeans(n_clusters=2)
# Training classifier
clf.fit(X)

# Calculating accuracy
correct = np.sum(clf.labels_ == y)

print(correct/len(X)*100)
