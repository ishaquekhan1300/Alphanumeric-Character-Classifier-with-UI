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
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import turtle
import skimage.io as ski_io

np.set_printoptions(threshold = sys.maxsize)

def shuffle_together(a, b):
    s = np.random.permutation(len(a))
    return a[s], b[s]

def drawing(x, y):
    tortimer.ondrag(None)
    tortimer.setheading(tortimer.towards(x, y))
    tortimer.goto(x, y)
    tortimer.ondrag(drawing)

def snap(x, y):
    tortimer.penup()
    tortimer.goto(x, y)
    tortimer.pendown()


def reset():
    tortimer.clear()

def save():
    print("Drawing Saved")
    data = window.getcanvas()
    data.postscript(file = "eval.eps", colormode = "mono", pageheight = 31, pagewidth = 31)

map = {}
with open('symbols.csv', newline = '') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        map[row['symbol_id']] = row['latex']

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

model = models.Sequential()

model.add(layers.Conv2D(16, (5, 5), activation = 'relu', input_shape = (32, 32, 1)))
model.add(layers.Dropout(0.25))
model.add(layers.MaxPooling2D( (2, 2) ))
model.add(layers.Conv2D(32, (5, 5), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(116, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(all_data.reshape(4658, 32, 32, 1), to_categorical(all_labels), epochs = 10, batch_size = 8)

print("Final model trained")

window = turtle.Screen()
window.setup(height = 640, width = 640)
tortimer = turtle.Turtle()
tortimer.speed(0)
tortimer.pensize(12)
tortimer.ondrag(drawing)
window.onkey(reset, "Escape")
window.onkey(save, "s")
window.onclick(snap, 1)
window.listen()
window.mainloop()
read_image = ski_io.imread("eval.eps")
plt.imshow(read_image)
plt.show()
read_image = read_image.astype('float32') / 255
read_image = read_image[:, :, 0]
read_image = read_image.reshape(1, 32, 32, 1)
prediction = np.argmax(model.predict(read_image))
print(map[str(prediction)])