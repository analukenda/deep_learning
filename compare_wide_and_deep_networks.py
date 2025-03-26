from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense
import time

from matplotlib import pyplot as plt

num_classes = 10
num_epochs = 10

# učitavanje MNIST baze podataka
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normaliziranje podataka na vrijednosti 0 - 1
x_train = x_train / 255
x_test = x_test / 255

variable_width = 300
wide_model_accuracy = []
wide_model_time = []

# stvaranje modela raznih sirina
for i in range(1,6):
    accuracy = 0.0
    duration = 0.0
    # vise pokretanja radi nasumicnosti procesa
    for j in range(3):
        # stvaranje modela
        model = Sequential()
        model.add(Flatten(input_shape=([28, 28])))
        model.add(Dense(variable_width * i, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # konfiguiranje modela
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # treniranje modela
        start=time.time()
        epochs = model.fit(x_train, y_train, epochs=num_epochs, batch_size=24, validation_data=(x_test, y_test))
        end=time.time()
        accuracy+=epochs.history['accuracy'][-1]
        duration += (end - start)

    #spremanje prosjecnih podataka
    wide_model_accuracy.append(accuracy/3)
    wide_model_time.append(duration/3)

print('Tocnosti sirokih modela raznih sirina:', wide_model_accuracy)
print('Vremena treniranja sirokih modela raznih sirina: ', wide_model_time)

# Crtanje grafa točnosti
plt.plot(range(variable_width, variable_width * 6, variable_width), wide_model_accuracy, 'r', label='Točnost generalizacije')
plt.title('Promjena točnosti generalizacije')
plt.xlabel('Širina')
plt.ylabel('Točnost generalizacije')
plt.legend()
plt.show()

# Crtanje grafa točnosti
plt.plot(range(variable_width, variable_width * 6, variable_width), wide_model_time, 'g', label='Vrijeme treniranja')
plt.title('Promjena vremena treniranja')
plt.xlabel('Širina')
plt.ylabel('Vrijeme treniranja')
plt.legend()
plt.show()

best_wide_width, best_wide_score = max(enumerate(wide_model_accuracy), key=lambda x:x[1])
print('Najbolju tocnost postize mreza sirine ' + str((best_wide_width + 1) * variable_width) + ': ' + str(best_wide_score))

variable_depth = 3
fixed_width = 60
deep_model_accuracy = []
deep_model_time =[]

# stvaranje modela raznih sirina
for i in range(1, 6):
    accuracy = 0.0
    duration = 0.0
    # vise pokretanja radi nasumicnosti procesa
    for j in range(3):
        # stvaranje modela
        model = Sequential()
        model.add(Flatten(input_shape=([28, 28])))
        for j in range(variable_depth * i):
            model.add(Dense(fixed_width, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # konfiguiranje modela
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # treniranje modela
        start = time.time()
        epochs = model.fit(x_train, y_train, epochs=num_epochs, batch_size=24, validation_data=(x_test, y_test))
        end = time.time()
        accuracy += epochs.history['accuracy'][-1]
        duration += (end - start)

    # spremanje prosjecnih podataka
    deep_model_accuracy.append(accuracy / 3)
    deep_model_time.append(duration / 3)

print('Tocnosti dubokih modela raznih dubina:', deep_model_accuracy)

# Crtanje grafa točnosti
plt.plot(range(variable_depth, variable_depth * 6, variable_depth), deep_model_accuracy, 'b', label='Točnost generalizacije')
plt.title('Promjena točnosti generalizacije')
plt.xlabel('Dubina')
plt.ylabel('Točnost generalizacije')
plt.legend()
plt.show()

best_deep_depth, best_deep_score = max(enumerate(deep_model_accuracy), key=lambda x:x[1])
print('Najbolju tocnost postize mreza dubine ' + str((best_deep_depth + 1) * variable_depth) + ': ' + str(best_deep_score))



