"""
***********************************************
MACHINE LEARNING - Support Vector Machines - Palmer Penguin Data set
***********************************************
Author: Adam Gużewski

Input data: penguin.csv

Link to data set: https://cloud.r-project.org/web/packages/palmerpenguins/index.html

To run the program you should type in terminal: python main.py

The data set consists of 344 examples of penguins.

My program is going to analyze the Palmer Penguins data set and train
Support Vector Machine Classifier to classify the data

"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics, svm
import seaborn as sns

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

data_filename = 'penguins_data.csv'

penguins = pd.read_csv(data_filename)

print(penguins)

print(penguins.info())


# In .info() function I found that there is some data missing, I am going to filter the data set

# filtered_penguins = penguins['culmen_length_mm'] > 1
# penguins = penguins[filtered_penguins]
# print(penguins.head())

# I want to remove the NaN values for better results and no warnings

print('Average culmen length: ', penguins['culmen_length_mm'].mean())
print('Average culmen depth: ', penguins['culmen_depth_mm'].mean())
print('Average flipper lenght: ', penguins['flipper_length_mm'].mean())
print('Average body mass: ', penguins['body_mass_g'].mean())

penguins.loc[penguins.culmen_length_mm.isnull(), 'culmen_length_mm'] = 43.9
penguins.loc[penguins.culmen_depth_mm.isnull(), 'culmen_depth_mm'] = 17.1
penguins.loc[penguins.flipper_length_mm.isnull(), 'flipper_length_mm'] = 200.9
penguins.loc[penguins.body_mass_g.isnull(), 'body_mass_g'] = 4201.7
penguins.loc[penguins.sex.isnull(), 'sex'] = 'FEMALE'

# penguins['sex'].replace(['MALE', 'FEMALE'], [0, 1], inplace=True)
# penguins['island'].replace(['Torgersen', 'Biscoe', 'Dream'], [0, 1, 2], inplace=True)
# penguins['species'].replace(['Adelie', 'Chinstrap', 'Gentoo'], [1, 2, 3], inplace=True)

# There are none NaN values anymore

print(penguins.isnull().sum())

print(penguins.corr())

# Creating a pairplot of the dataset

sns.pairplot(penguins, hue='species', palette='Dark2')
plt.savefig('pairplot.png')

# Creating a heatmap to find best correlation

plt.figure(figsize=(8, 6))
sns.heatmap(penguins.corr(), annot=True)
plt.savefig('heatmap.png')

# The best correlation is for body_mass_g & flipper_length_mm

print(penguins.shape)

# Splitting the data into a training and testing set

train, test = train_test_split(penguins, test_size=0.2, random_state=5)
print('Size of training data: ', train.shape)
print('Size of testing data: ', test.shape)

print(train)

X_train = train[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y_train = train.species

X_test = test[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y_test = test.species

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

prediction = model.predict(X_test)

# Printing confusion matrix for First training model

print('\n\nConfusion Matrix For: ')
print(confusion_matrix(y_test, prediction))

# Printing classification report for First training model

print('\n\nClassification Report:')
print(classification_report(y_test, prediction, zero_division=0))

print('Accuracy of the SVM is: ', metrics.accuracy_score(prediction, y_test))

plt.show()
