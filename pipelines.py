# Reworking Data Preparation process with transformers and sklearn pipelines
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Pipeline
from sklearn.pipeline import Pipeline

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
data = pd.read_csv('Data/train.csv')

features = data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
labels = data['Survived']

# Transformer to handle dataframes
class frameToArray(BaseEstimator, TransformerMixin):
  def __init__(self, columns):
    self.columns = columns
  
  def fit(self, X):
    return self

  def transform(self, X):
    return np.array(X[self.columns]).reshape(-1,1)


catPipeline = Pipeline([
  ('array', frameToArray('Pclass')),
  ('encoding', OneHotEncoder())
])

test = catPipeline.fit_transform(features).toarray()

print(test)