#Student Performance Classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
dataset["Class"] = dataset["Class"].map({"H":2,"M":1,"L":0})
#Dropping Irrelevant Features
feature_drop = ['NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester']
dataset = dataset.drop(feature_drop,axis=1)

#Setting Test and Target Variables
X = dataset.iloc[:,[0,1,2,3,4,5,6,7,8]].values
y = dataset.iloc[:,[9]].values

#Splitting into Training and Test Data
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, train_size = 0.75)


