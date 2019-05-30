# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:59:48 2019

@author: iasedric
"""

# Importing the libraries
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
y = dataset.iloc[:, 30:31]

X = dataset.iloc[:, 1:30]

X_resample, y_resample = SMOTE().fit_sample(X,y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_resc = sc.transform(X)



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',max_depth = 6 , random_state = 0)
classifier.fit(X_train, y_train)


#Applying the Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#parameters = [{'n_estimators' : [90,92,94,96,98,100,102,104,106,108,110],
#                'max_depth' : [8,9,10,11,12]
#                }]

parameters = [{'n_estimators' : [37,38,39],
                'max_depth' : [9,10]
                }]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1)


grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators = best_parameters['n_estimators'], criterion = 'entropy',max_depth = best_parameters['max_depth'] , random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_Test_proba = classifier.predict(X_test)
y_pred_test = (y_pred_Test_proba > 0.5)

y_pred_Train_proba = classifier.predict(X_train)
y_pred_train = (y_pred_Train_proba > 0.5)

CAP = np.stack((y_train,y_pred_Train_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_train' : CAP[:,0],
                    'y_train_proba' : CAP[:,1]}).sort_values(by=['y_train_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Fraud\For_CAP_wRegressionForest.csv", index=False, encoding='utf_8_sig')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF_test = confusion_matrix(y_test, y_pred_test)


CAP = np.stack((y_test,y_pred_Test_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_test' : CAP[:,0],
                    'y_test_proba' : CAP[:,1]}).sort_values(by=['y_test_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Fraud\For_CAP_wRegressionForest_test.csv", index=False, encoding='utf_8_sig')

# Making the Confusion Matrix

cm_RF_train = confusion_matrix(y_train, y_pred_train)


y_pred_proba = classifier.predict(X_resc)
y_pred = (y_pred_proba > 0.5)
cm_RF = confusion_matrix(y, y_pred)

