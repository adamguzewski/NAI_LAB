"""
***********************************************
MACHINE LEARNING - Support Vector Machines -  Fashion-MNIST / Zalando Dataset
Neural network
***********************************************
Author: Adam Gużewski

Dataset source: tf.keras.datasets.fashion_mnist

To run the program you should type in terminal: python main.py

The data set consists of 60 000 learning data and 10 000 of testing data

My program is going to analyze the  Fashion-MNIST data set and use neural network to classify the clothes

"""

# Importing libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, add, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Data loading
fashion_data = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_val, y_val) = fashion_data.load_data()

# the figure is a matrix with RGB value from 0 to 255

plt.figure(figsize=(7, 7))
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()


def plot_digit(digit, dem=28, font_size=7):
    """
    Function will return the matrix with the values of RGB for each pixel (scale of gray_
    :param digit:
    :param dem: size of element
    :param font_size: size of font in pixels
    :return: image with values of RGB for each pixel
    """
    max_ax = font_size * dem

    fig = plt.figure(figsize=(10, 10))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')

    for idx in range(dem):
        for jdx in range(dem):
            t = plt.text(idx * font_size, max_ax - jdx * font_size, digit[jdx][idx], fontsize=font_size,
                         color="#000000")
            c = digit[jdx][idx] / 255.
            t.set_bbox(dict(facecolor=(c, c, c), alpha=0.5, edgecolor='#f2f2f2'))
    plt.savefig('Example_of_element_representation.png')
    plt.show()


plot_digit(X_train[11])

# names of dataset

names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot will show firs 100 elements of dataset

plt.figure(figsize=(33, 33))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(names[y_train[i]])
plt.savefig('First_100_elements_of_dataset.png')
plt.show()

# to prepare data I will divide my values with 255 to have the lowest possible values

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Now I'm going to prepare the data which needs to be predicted by model

y_train = to_categorical(y_train, len(names))
y_val = to_categorical(y_val, len(names))

# preparing training model

model = Sequential()

# adjusting model to analyse one vector 1x784

model.add(Flatten(input_shape=(28, 28)))

# Firstly I chose only one hidden layer - Dense

model.add(Dense(128, activation='relu'))

# In this problem I have to predict 10 classes that is why I need 10 neurons
# I chose softmax because the output is going to be an array with 10 values which is going to be probability

model.add(Dense(10, activation='softmax'))

# choosing adam for the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

# I added EarlyStop to prevent model from overtrain

EarlyStop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

model.summary()

# Training model

history = model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=256, validation_data=(X_val, y_val), callbacks=[EarlyStop])

# Creating the plots of loss and accuracy

epochs = len(history.history['loss'])

y1 = history.history['loss']
x = np.arange(1, epochs + 1)

plt.plot(x, y1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('Epochs_Loss.png')
plt.show()

y1 = history.history['accuracy']
x = np.arange(1, epochs + 1)

plt.plot(x, y1)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.tight_layout()
plt.savefig('Epochs_acc.png')
plt.show()

# Prediction - I am going to check if our model predicts correctly

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)


def plot_value_img(i, predictions, true_label, img):
    """
    Function is going to check if prediction is working correctly
    :param i: number of element
    :param predictions: predicted element
    :param true_label: true label for chosen i element
    :param img: picture of predicted element
    :return: shows plot of classes, green for good prediction, red for bad one
    """
    predictions, true_label, img = predictions[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions)
    true_value = np.argmax(true_label)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    plt.yticks(np.arange(len(names)), names)
    thisplot = plt.barh(range(10), predictions, color='gray')
    thisplot[predicted_label].set_color('r')
    thisplot[true_value].set_color('g')

    plt.subplot(1, 2, 2)

    plt.imshow(img, cmap=plt.cm.binary)
    if predicted_label == true_value:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(names[predicted_label],
                                         100 * np.max(predictions),
                                         names[true_value]), color=color)
    plt.show()


plot_value_img(6543, y_val_pred, y_val, X_val)
