#Student Performance Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Import Dataset
dataset = pd.read_csv("xAPI-Edu-Data.csv")
#X = dataset.iloc[:,[0,8,9,10,11,12,13,14,15]].values
#y = dataset.iloc[:,[16]].values

#Data Cleaning
#Encoding Gender Feature
dataset["gender"] = dataset["gender"].map({"M":1,"F":0})
#Encoding Guardian Relationship
dataset["Relation"] = dataset["Relation"].map({"Father":1,"Mum":0})
#Encoding Survey Results
dataset["ParentAnsweringSurvey"] = dataset["ParentAnsweringSurvey"].map({"Yes":1,"No":0})
#Encoding Parents Satisfaction
dataset["ParentschoolSatisfaction"] = dataset["ParentschoolSatisfaction"].map({"Good":1,"Bad":0})
#Encoding Absent Days of the Student
dataset["StudentAbsenceDays"] = dataset["StudentAbsenceDays"].map({"Under-7":1,"Above-7":0})
#Encoding Final Assesment of students
dataset["Class"] = dataset["Class"].map({"H":1,"M":1,"L":0})
#Dropping Irrelevant Features
feature_drop = ['NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester']
dataset = dataset.drop(feature_drop,axis=1)

#Setting Test and Target Variables
X = dataset.iloc[:,[0,1,2,3,4,5,6,7,8]].values
y = dataset.iloc[:,[9]].values

#One hot encoding the multi class attribute
#encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_Y)

#Splitting into Training and Test Data
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, train_size = 0.75)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating Neural Network Model
model = Sequential()
model.add(Dense(12, input_dim=9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=0)

#Prdicting New Inputs
y_pred= model.predict_classes(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculating Accuracy
correct_pred_sum = 0
for i in range(cm.shape[1]):
    correct_pred_sum += cm[i][i]
print("Accuracy = %.2f%%"%(((correct_pred_sum/120)*100)))




