# -*- coding: utf-8 -*-


"""A/B testing with Machine learning code.ipynb
## A/B Testing with Machine Learning
Machine Learning enables modelling of complex systems unlike the statistical inference approach.

Feature significance is what tells whether the experiment had some impact and also the contribution of other features.

## Data
The BIO data for this project is a “Yes” and “No” response of online users to the following question:


`Q: Do you know the brand SmartAd?`

      Yes
      No
The data has the following columns:
  - "auction_id": the unique id of the online user who has been presented the BIO.
  - "experiment": which group the user belongs to - control or exposed.
  - "date": the date in YYYY-MM-DD format
  - "hour": the hour of the day in HH format.
  - "device_make": the name of the type of device the user has e.g. Samsung
  - "platform_os": the id of the OS the user has.
  - "browser": the name of the browser the user uses to see the BIO questionnaire.
  - "yes": 1 if the user chooses the “Yes” radio button for the BIO questionnaire.
  - "no": 1 if the user chooses the “No” radio button for the BIO questionnaire.
"""

#imoprtant liberaries
import os
import warnings
import sys

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from scipy.stats import skew, norm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier


#read an csv file
df = pd.read_csv(r'C:\Users\Gezahegne\10-Accademy\week-20\AB-Testing\data\AdSmartABdata.csv')
df.head(10)

#missing value check
df.isna().any()

#classify columns by datatype and make list
#data type =object
categorical = ['auction_id', 'experiment', 'date', 'device_make', 'browser']
#data type ='int16', 'int32', 'int64', 'float16', 'float32', 'float64
numerical = ['hour', 'platform_os', 'yes', 'no']

features = categorical + numerical 
df1 = df[features]
df1.head(3)


def is_outlier(data,col):
    z_scores = stats.zscore(data[col])
    abs_z_scores = np.abs(z_scores)
    if (abs_z_scores < 3). all():
        print('there is  outlier')
    else:
        print('there is no outlier')
        
#is_outlier(df,col) 

# Find skewed numerical features
#result showes us except hour data hass highly skewed features
skew_features = df1[numerical].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features


# get the location of the 3 categorical columns
features = df.copy()
indices = []
for col in ['browser', 'experiment', 'device_make']:
    k = features.columns.get_loc(col)
    indices.append(k)
    
indices
 
# Encoding categorical variables using Label Encoder
columns = indices
for col in columns:
    x = features.iloc[:, col].values
    x = x.reshape(-1,1)
    encoder = LabelEncoder()
    encoder = encoder.fit(x)
    x = encoder.transform(x)
    features.iloc[:, col] = x 

# features = pd.get_dummies(df)
print(features.shape)
features.head()

# create the target variable from the yes/no cols then drop yes/no cols

# the 1st in yes remain the same, the 1st in no become 2s, the entries with 0s in both cols remain as 0s.
features['target'] = 0

features = features[features.target != 0]
features.loc[features['target'] ==2, 'target'] = 0
print(features.shape)
features.target.value_counts()

features.head()
#dependent variable is target
# dependent and independent variables
x = features.drop(['target'], axis = 1)
y = features[['target']]



#  split dataset to train and test sets (90:10)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .1, random_state = 0)
print('x train', x_train.shape)
print('y train', y_train.shape)
print('x test', x_test.shape)
print('y test', y_test.shape)

# get the validation set from the train set (70:20)

# the % changes to 22 to be representative of the 20 expected originally
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = .22, random_state = 0)
print('x train', x_train.shape)
print('y train', y_train.shape)
print('x validation', x_val.shape)
print('y validation', y_val.shape)
print('x test', x_test.shape)
print('y test', y_test.shape)


     ##linear regration model ###
with mlflow.start_run():
     myreg=LinearRegression()
     myreg.fit(x_train,x_test)
     myreg.score(x_train,x_test)

# feature importance
feat_imp_dict = dict(zip(x_train.columns, myreg.coef_[0]))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp.sort_values(by=['FeatureImportance'], ascending=False)

# feature weights for every class
coef_0=myreg.coef_[0]
coef_1=myreg.coef_[1]
coef_2=myreg.coef_[2]
print(coef_0)
print(coef_1)
print(coef_2)

##XGB
xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
colsample_bynode=1, colsample_bytree=1, gamma=0,
importance_type='gain', learning_rate=0.1, max_delta_step=0,
max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
silent=None, subsample=1, verbosity=1)
       
xgbr.fit(xtrain, ytrain)
#After training the model, we'll check the model training score.

score = xgbr.score(x_train, y_train)  
print("Training score: ", score)
#We can also apply the cross-validation method to evaluate the training score.

scores = cross_val_score(xgbr, x_train, y_train,cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

#can predict test data, then check the prediction accuracy. 
#Here, we'll use MSE and RMSE as accuracy metrics.

y_pred = xgbr.predict(x_test)
mse = mean_squared_error(ytest, y_pred)
print("MSE: %.2f" % mse)
MSE: 3.35
print("RMSE: %.2f" % (mse**(1/2.0)))

#visualize the original and predicted test data in a plot to compare visually.

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title(" test and predicted data")
plt.legend()
plt.show()

##Decision Trees##
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

scores = cross_val_score(estimator = tree, X = x_train, y = y_train, cv = 5)
print(scores)
print("mean decision trees score : ", scores.mean())
# feature importance
feat_importance = tree.tree_.compute_feature_importances(normalize=False)
feat_imp_dict = dict(zip(x_train.columns, tree.feature_importances_))
feat_imp_1 = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp_1.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp_1.sort_values(by=['FeatureImportance'], ascending=False).head()
#visualize
plt.figure(figsize = (6,4))
sb.barplot(y = feat_imp_1.FeatureImportance, x = feat_imp_1.index)
plt.title('Feature Importances in Decision Trees')
plt.xticks(rotation = 45)
# using Decision Tree to run predictions on x_test
y_pred = tree.predict(x_test)
a = pd.DataFrame(y_pred)
a.columns = ['pred']
a.pred.value_counts()