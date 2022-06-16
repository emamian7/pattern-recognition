import numpy
import tensorflow as tf
from tensorflow import keras
max_review_length = 1600
top_words = 10000


# Using keras to load the dataset with the top_words
def import_dataset():
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
    return X_train, X_test, y_train, y_test


# Pad the sequences to the same length
def preprocess_data(X_train, X_test):
    new_X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
    new_X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)
    return new_X_train, new_X_test


def create_model():
    # Using embedding from Keras
    embedding_vecor_length = 300
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

    model.add(keras.layers.Convolution1D(64, 3, padding='same'))
    model.add(keras.layers.Convolution1D(32, 3, padding='same'))
    model.add(keras.layers.Convolution1D(16, 3, padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(180,activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1,activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train):
    # Log to Tensorboard
    tensorBoardCallback = keras.callbacks.TensorBoard(log_dir='./logs', write_graph=True)
    model.fit(X_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)


def evaluate_model(model, X_test, y_test):
    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def run():
    X_train, X_test, y_train, y_test = import_dataset()
    X_train, X_test = preprocess_data(X_train, X_test)
    model = create_model()
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)


run()
