"""
***********************************************
MACHINE LEARNING - Support Vector Machines - Wheat Seeds Data set
***********************************************
Author: Adam Gużewski

Input data: seeds_dataset.txt

Link to data set: https://archive.ics.uci.edu/ml/datasets/seeds

To run the program you should type in terminal: python main.py

The data set consists of 209 examples of wheat seeds.

My program is going to analyze the wheat seeds data set and train
Support Vector Machine Classifier to classify the data

"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn.svm import SVC

# Adding column labels

columns_names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel',
                 'Width of kernel', 'Asymmetry coefficient',
                 'Length of kernel groove', 'Class']

# Importing the data set

data_filename = 'seeds_dataset.txt'
seeds = pd.read_csv(data_filename, delim_whitespace=True, names=columns_names)

print(seeds)

# Creating a pairplot of the dataset

sns.pairplot(seeds, hue='Class', palette='Dark2')
plt.savefig('pairplot.png')

# Building and showing the correlation between the dataset features

correlation = seeds.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(correlation, annot=True, square=True, ax=ax)
plt.yticks(rotation=360)
plt.savefig('heatmap.png')

# Visualisation of kdeplot

class_two = seeds[seeds["Class"] == 2]
sns.kdeplot(x=class_two['Asymmetry coefficient'], y=class_two['Compactness'], cmap="plasma", shade=True)
plt.show()
plt.savefig('kdeplot.png')

# Splitting the data into a training and testing set

X = seeds.drop('Class', axis=1)
y = seeds['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)# Calling the SVC model from sklearn and fitting the model to the training data

svc_model = SVC()

svc_model.fit(X_train, y_train)

# Getting predictions from the model

predictions = svc_model.predict(X_test)

# printing confusion matrix and classification report

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

# Printing the accuracy score

print('The accuracy of SVM is:', metrics.accuracy_score(predictions, y_test))

plt.show()
