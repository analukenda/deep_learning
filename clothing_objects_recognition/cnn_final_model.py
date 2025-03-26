import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Rescaling, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import to_categorical


def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_classes = y_test.shape[1]
    return (x_train, y_train), (x_test, y_test), num_classes


def build_model(num_classes):
    model = Sequential()
    model.add(Rescaling(1. / 255, input_shape=(28, 28, 1)))

    model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1), activation='swish'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='swish'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(12, (3, 3), activation='swish'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(strides=(2, 2)))

    model.add(Dense(128, activation='swish'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(24, (3, 3), activation='swish'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    epochs = 20
    (x_train, y_train), (x_test, y_test), num_classes = load_data()
    model = build_model(num_classes)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=500)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # Vizualizacija promjene točnosti kroz epohe
    plt.plot(range(1, epochs + 1), history.history['accuracy'], color='blue')
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], color='orange')
    plt.legend(['Točnost treniranja', 'Točnost testiranja'])
    plt.xlabel('Epoha')
    plt.ylabel('Točnost')
    plt.title('Promjena točnosti kroz epohe')
    plt.show()

    # Vizualizacija promjene gubitka kroz epohe
    plt.plot(range(1, epochs + 1), history.history['loss'], color='blue')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], color='orange')
    plt.legend(['Gubitak treniranja', 'Gubitak testiranja'])
    plt.xlabel('Epoha')
    plt.ylabel('Gubitak')
    plt.title('Promjena gubitka kroz epohe')
    plt.show()


if __name__ == '__main__':
    main()
