"""
***********************************************
MACHINE LEARNING - Support Vector Machines - Palmer Penguin Data set
Neural network
***********************************************
Author: Adam Gużewski

Input data: penguin_data.csv

Link to data set: https://cloud.r-project.org/web/packages/palmerpenguins/index.html

To run the program you should type in terminal: python main.py

The data set consists of 344 examples of penguins.

My program is going to analyze the Palmer Penguins data set and use neural network to classify the data

"""

# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# I want to see all table
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

data_filename = 'penguins_data.csv'

penguins = pd.read_csv(data_filename)

# print(penguins)

# I want to remove the NaN values for better results and no warnings

# print('Average culmen length: ', penguins['culmen_length_mm'].mean())
# print('Average culmen depth: ', penguins['culmen_depth_mm'].mean())
# print('Average flipper lenght: ', penguins['flipper_length_mm'].mean())
# print('Average body mass: ', penguins['body_mass_g'].mean())

penguins.loc[penguins.culmen_length_mm.isnull(), 'culmen_length_mm'] = 43.9
penguins.loc[penguins.culmen_depth_mm.isnull(), 'culmen_depth_mm'] = 17.1
penguins.loc[penguins.flipper_length_mm.isnull(), 'flipper_length_mm'] = 200.9
penguins.loc[penguins.body_mass_g.isnull(), 'body_mass_g'] = 4201.7
penguins.loc[penguins.sex.isnull(), 'sex'] = 'FEMALE'
penguins.loc[(penguins.sex == '.'), 'sex'] = 'FEMALE'

# Plots for all columns

sns.countplot(x=penguins['species'])
plt.tight_layout()
# plt.show()
sns.countplot(x=penguins['island'])
plt.tight_layout()
# plt.show()
sns.displot(x=penguins['culmen_length_mm'], bins=40)
plt.tight_layout()
# plt.show()
sns.displot(x=penguins['culmen_depth_mm'], bins=40)
plt.tight_layout()
# plt.show()
sns.displot(x=penguins['flipper_length_mm'], bins=40)
plt.tight_layout()
# plt.show()
sns.displot(x=penguins['body_mass_g'], bins=40)
plt.tight_layout()
# plt.show()
sns.countplot(x=penguins['sex'])
plt.tight_layout()
# plt.show()

# I'm checking if sex and island matters in results

pd.crosstab(penguins['island'], penguins['species']).plot.bar()
plt.tight_layout()
# plt.show()

pd.crosstab(penguins['sex'], penguins['species']).plot.bar()
plt.tight_layout()
# plt.show()

# It seems the sex of the penguins does not matter to the results, so I am removing the sex column

penguins = penguins.drop('sex', axis=1)
# print(penguins.head())

# Creating a pairplot of the penguins dataset

sns.pairplot(penguins, hue='species', palette='Dark2')
plt.savefig('pairplot.png')

# Creating a heatmap to find best correlation

plt.figure(figsize=(10, 8))
sns.heatmap(penguins.corr(), annot=True)
plt.savefig('heatmap.png')

# Preparing the dataset for training

penguins['culmen_area_mm'] = penguins['culmen_length_mm'] * penguins['culmen_depth_mm']
# print(penguins.head())

penguins = pd.concat([penguins, pd.get_dummies(penguins['island'])], axis=1)
# print(penguins.head())

penguins = penguins.drop('island', axis=1)
# print(penguins.head())

species = penguins['species'].value_counts()
species_dict = {species: idx for idx, species in enumerate(list(species.index))}
print(species_dict)

penguins['species'] = penguins['species'].map(species_dict)
# print(penguins.head())

sns.heatmap(penguins.corr(), annot=True)
plt.savefig('heatmap_without_names.png')
plt.show()

# data mixing

penguins = penguins.sample(frac=1).reset_index(drop=True)
# print(penguins)

labels = penguins.pop('species')
# print(labels[: 10])

penguins = (penguins - penguins.min()) / (penguins.max() - penguins.min())
# print(penguins)

all_penguins = penguins.values
# print(all_penguins)

# Splitting the data into a training and testing set

train_size = round(0.8 * all_penguins.shape[0])

train_data = all_penguins[: train_size]
test_data = all_penguins[train_size:]

train_labels = labels[: train_size]
test_labels = labels[train_size:]

assert (len(train_data) == len(train_labels))


# Training our neural network

def build_model():
    model = keras.models.Sequential([
        keras.layers.Dense(64, 'selu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(3, 'softmax'),
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

# I have tested that adam optimizer gives good results


my_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

all_values_loss_history = []
all_values_scores = []

skf = StratifiedKFold(n_splits=3)

# splitting the data into 3 parts

for train_index, test_index in skf.split(train_data, train_labels):
    print('Training...')
    x_train, x_val = train_data[train_index], train_data[test_index]
    y_train, y_val = train_labels[train_index], train_labels[test_index]

    model = build_model()

    history = model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val),
                        callbacks=[my_callback], verbose=1)
    all_values_loss_history.append(history.history['loss'])
    all_values_scores.append(model.evaluate(x=x_val, y=y_val, verbose=0))

average_loss = np.mean([x[0] for x in all_values_scores])
print(f'\nAverage loss is: {average_loss}')

average_values_loss_history = [np.mean([x[i] for x in all_values_loss_history]) for i in range(26)]
print(len(average_values_loss_history))

plt.plot(range(1, len(average_values_loss_history) + 1), average_values_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

model = build_model()

history = model.fit(x=train_data, y=train_labels, epochs=26, callbacks=[my_callback], verbose=1)

epochs = len(history.history['loss'])

y1 = history.history['loss']
x = np.arange(1, epochs + 1)

plt.plot(x, y1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('Epochs_Loss.png')
plt.show()

y1 = history.history['acc']
x = np.arange(1, epochs + 1)

plt.plot(x, y1)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.tight_layout()
plt.savefig('Epochs_acc.png')
plt.show()

# Printing the results

print('[Loss of all data,Accuracy]')
print(model.evaluate(test_data, test_labels))

# Trained model has at least 97% of accuracy
