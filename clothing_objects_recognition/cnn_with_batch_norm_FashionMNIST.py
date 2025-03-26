from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Učitavanje podataka
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# One-hot kodiranje oznaka primjeraka
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# Broj klasa = 10
num_classes = y_test.shape[1]
# Broj epoha
epochs = 40

# Neuronska mreža
model = Sequential()
# 1. konvolucijski sloj, 32 filtera dimenzija 3x3
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
# 1. sloj batch normalization
model.add(BatchNormalization())
# 1. sloj sažimanja (maksimalnom vrijednošću)
model.add(MaxPooling2D(strides=(2,2)))
# 2. sloj bacth normalization
model.add(BatchNormalization())
# 2. konv. sloj, 32 filtera 3x3
model.add(Conv2D(32, (3,3), activation='relu'))
# 2. sloj sažimanja
model.add(MaxPooling2D(strides=(2,2)))
# Splašnjavanje
model.add(Flatten())
# 1. Potpuno povezani sloj
model.add(Dense(128,activation='relu'))
# Dropout sloj
model.add(Dropout(rate=0.5))
# Izlazni sloj
model.add(Dense(num_classes, activation='softmax'))

# Konfiguracija modela
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treniranje modela
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200)

# Evaluacija modela
scores = model.evaluate(x_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

# Vizualizacija promjene točnosti kroz epohe
plt.plot(range(1,epochs+1),history.history['accuracy'],color='blue')
plt.plot(range(1,epochs+1),history.history['val_accuracy'],color='orange')
plt.legend(['Točnost treniranja','Točnost testiranja'])
plt.xlabel('Epoha')
plt.ylabel('Točnost')
plt.title('Promjena točnosti kroz epohe')
plt.show()

# Vizualizacija promjene gubitka kroz epohe
plt.plot(range(1,epochs+1),history.history['loss'],color='blue')
plt.plot(range(1,epochs+1),history.history['val_loss'],color='orange')
plt.legend(['Gubitak treniranja','Gubitak testiranja'])
plt.xlabel('Epoha')
plt.ylabel('Gubitak')
plt.title('Promjena gubitka kroz epohe')
plt.show()