import sys
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import regularizers
from keras.utils import to_categorical
import matplotlib
from matplotlib import pyplot as plt
import turtle
import skimage.io as ski_io

matplotlib.use("TkAgg")

np.set_printoptions(threshold = sys.maxsize)

def shuffle_together(a, b):
    s = np.random.permutation(len(a))
    return a[s], b[s]

all_data = np.load("alphanum-hasy-data-X.npy")
all_labels = np.load("alphanum-hasy-data-Y.npy")

all_data = all_data.astype('float32') / 255

all_data, all_labels = shuffle_together(all_data, all_labels)

train_samples =3000
validation_samples = 500

train_set = all_data[:train_samples]
train_set = train_set.reshape(3000, 32, 32, 1)
train_labels = to_categorical(all_labels[:train_samples])
validation_set = all_data[train_samples:validation_samples + train_samples]
validation_set = validation_set.reshape(500, 32, 32, 1)
validation_labels = to_categorical(all_labels[train_samples:validation_samples + train_samples])
test_set = all_data[3500:]
test_set = test_set.reshape(1158 , 32, 32, 1)
test_labels = to_categorical(all_labels[train_samples + validation_samples:])

median_trials = 0
mean_trials = 0
max_trials = 0
accuracy_trials = 0
trials = 10

for test_run in range(1, trials + 1):
    model = models.Sequential()
    model2 = models.Sequential()
    
    model.add(layers.Conv2D(16, (5, 5), activation = 'relu', input_shape = (32, 32, 1)))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D( (2, 2) ))
    model.add(layers.Conv2D(32, (5, 5), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(116, activation = 'softmax'))
    
    # Test model below, parameters were changed to find a configuration which maximized validation accuracy
    
    model2.add(layers.Conv2D(16, (5, 5), activation = 'relu', input_shape = (32, 32, 1)))
    model2.add(layers.Dropout(0.25))
    model2.add(layers.MaxPooling2D( (2,2) ))
    model2.add(layers.Conv2D(32, (5, 5), activation = 'relu'))
    model2.add(layers.Flatten())
    model2.add(layers.Dense(116, activation = 'softmax'))
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model2.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(train_set, train_labels, epochs = 10, batch_size = 8, validation_data = (validation_set, validation_labels))
    history2 = model2.fit(train_set, train_labels, epochs = 10, batch_size = 16, validation_data = (validation_set, validation_labels))
    
    #acc_vals = history.history['accuracy']
    val_acc_vals = history.history['val_accuracy']
    #acc_vals2 = history2.history['accuracy']
    val_acc_vals2 = history2.history['val_accuracy']
    #plt.plot(range(1, 11), acc_vals, 'bo', label = "Baseline T Accuracy")
    #plt.plot(range(1, 11), val_acc_vals, 'b', label = "Baseline V Accuracy")
    #plt.plot(range(1, 7), acc_vals2, 'ro', label = "Training Accuracy")
    #plt.plot(range(1, 7), val_acc_vals2, 'r', label = "Validation Accuracy")
    #plt.title("Training and Validation Accuracy")
    #plt.xlabel("Epochs")
    #plt.ylabel("Accuracy")
    #plt.legend()

    test_loss, test_acc = model.evaluate(test_set, test_labels)
    test_loss2, test_acc2 = model2.evaluate(test_set, test_labels)

    if (np.median(val_acc_vals2) > np.median(val_acc_vals)):
        median_trials += 1
    if (np.mean(val_acc_vals2) > np.mean(val_acc_vals)):
        mean_trials += 1
    if (np.max(val_acc_vals2) > np.max(val_acc_vals)):
        max_trials += 1
    if (test_acc2 > test_acc):
        accuracy_trials += 1

print("The trial model had a higher median validation accuracy " + str(median_trials) + " times out of " + str(trials) + " trials")
print("The trial model had a higher mean validation accuracy " + str(mean_trials) + " times out of " + str(trials) + " trials")
print("The trial model had a higher max validation accuracy " + str(max_trials) + " times out of " + str(trials) + " trials")
print("The trial model had a higher test accuracy " + str(accuracy_trials) + " times out of " + str(trials) + " trials")