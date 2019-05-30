# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:19:57 2019

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



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 29))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.5))

# Adding the third  hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred_Test_proba = classifier.predict(X_test)
y_pred_test = (y_pred_Test_proba > 0.5)


y_pred_Train_proba = classifier.predict(X_train)
y_pred_train = (y_pred_Train_proba > 0.5)


#CAP = np.stack((y_train,y_pred_Train_proba[:,1]), axis=1)

CAP = np.stack((y_train,y_pred_Train_proba[:,0]), axis=1)

CAP = pd.DataFrame({'y_train' : CAP[:,0],
                    'y_train_proba' : CAP[:,1]}).sort_values(by=['y_train_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Fraud\For_CAP_wANN.csv", index=False, encoding='utf_8_sig')


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ANN_test = confusion_matrix(y_test, y_pred_test)


CAP = np.stack((y_test,y_pred_Test_proba[:,0]), axis=1)

CAP = pd.DataFrame({'y_test' : CAP[:,0],
                    'y_test_proba' : CAP[:,1]}).sort_values(by=['y_test_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Fraud\For_CAP_wANN_test.csv", index=False, encoding='utf_8_sig')


# Making the Confusion Matrix

cm_ANN_train = confusion_matrix(y_train, y_pred_train)



y_pred_proba = classifier.predict(X_resc)
y_pred = (y_pred_proba > 0.5)

# Making the Confusion Matrix
cm_ANN = confusion_matrix(y, y_pred)



