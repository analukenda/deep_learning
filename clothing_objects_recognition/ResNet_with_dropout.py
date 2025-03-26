from keras.datasets import fashion_mnist
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np
import cv2

# Učitavanje podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Željene dimenzije ulaznih slika
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNEL = 3

# Iz 1 kanala u 3 kanala
x_train=np.repeat(x_train[...,np.newaxis],3,-1)
x_test=np.repeat(x_test[...,np.newaxis],3,-1)

# Stvaranje ulaznih podataka zeljene velicine
resized_train = np.ndarray(shape=(x_train.shape[0],IMG_HEIGHT,IMG_WIDTH,CHANNEL))
resized_test = np.ndarray(shape=(x_test.shape[0],IMG_HEIGHT,IMG_WIDTH,CHANNEL))
for input in range(x_train.shape[0]):
    resized_train[input] = cv2.resize(x_train[input], (IMG_HEIGHT,IMG_WIDTH),interpolation = cv2.INTER_AREA)
for input in range(x_test.shape[0]):
    resized_test[input] = cv2.resize(x_test[input], (IMG_HEIGHT,IMG_WIDTH) ,interpolation = cv2.INTER_AREA )

# One-hot kodiranje oznaka primjeraka
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# Broj klasa = 10
num_classes = y_test.shape[1]

# Ucitavanje predtreniranog ResNet50
imported_model= ResNet50(include_top=False,weights='imagenet',input_shape=(32,32,3),pooling='avg',classes=num_classes)
for layer in imported_model.layers:
    layer.trainable=False

dropout = 0.5

# Neuronska mreža
model = Sequential()
# 1. sloj, predtrenirani ResNet
model.add(imported_model)
# Splašnjavanje
model.add(Flatten())


# 1. potpuno povezani sloj
model.add(Dense(512,activation='relu'))
model.add(Dropout(dropout))

# 2. potpuno povezani sloj
model.add(Dense(128,activation='relu'))


# 3. potpuno povezani sloj
model.add(Dense(64,activation='relu'))



# Izlazni sloj
model.add(Dense(num_classes, activation='softmax'))

# Konfiguracija modela
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treniranje modela
history = model.fit(resized_train, y_train, validation_data=(resized_test, y_test), epochs=30, batch_size=200)

# Evaluacija modela
scores = model.evaluate(resized_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

# Prikazivanje grafikon gubitka tijekom treninga i validacije
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()