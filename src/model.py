import keras.utils as ku
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator

from .metrics import perplexity_raw
from .utils import train_test_split


class BaseNetwork(BaseEstimator):
    def __init__(self, tokenizer=Tokenizer()):

        self.tokenizer = tokenizer

    def etl(self, data):

        # basic cleanup
        corpus = data.lower().split("\n")

        # tokenization
        self.tokenizer.fit_on_texts(corpus)
        self.total_words = len(self.tokenizer.word_index) + 1

        # create input sequences using list of tokens
        input_sequences = []
        for line in corpus:
            # TODO: Probar con fastText y HashingVectorizer y los demás de text de keras.
            # Onehot no hace falta si usamos embedding
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                input_sequences.append(token_list[: i + 1])

        # pad sequences
        self.max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(
            pad_sequences(
                input_sequences, maxlen=self.max_sequence_len, padding="pre", value=0
            )
        )
        # create X and y
        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = ku.to_categorical(y, num_classes=self.total_words)

        return train_test_split(X, y)

    def generate_text(self, seed_text, next_words):

        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.max_sequence_len - 1, padding="pre"
            )
            predicted = self.net.predict_classes(token_list, verbose=0)

            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text

    def fit(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        arch="Baseline",
        earlystop=None,
        epochs=200,
        batch_size=None,
        verbose=2,
        activation="softmax",
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[perplexity_raw],
        hidden_lstm=1,
    ):

        self.net = Sequential()
        self.net.add(Embedding(self.total_words, 64, input_length=X_train.shape[1]))

        if arch == "Baseline":
            self.Baseline(activation=activation)
        elif arch == "LSTM_Embedding":
            self.LSTM_Embedding(hidden_lstm=hidden_lstm, activation=activation)
        else:
            raise Exception("Unknown network architecture")

        self.net.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print(self.net.summary())

        if earlystop is True:
            earlystop = [
                EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto"
                )
            ]
        self.net.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=None,
            verbose=verbose,
            callbacks=earlystop,
            validation_data=(X_test, y_test),
        )

    def Baseline(self, activation):
        """Simple network with an embedding layer and a dense one"""

        self.net.add(Flatten())
        self.net.add(Dense(self.total_words, activation=activation))

        return self

    def LSTM_Embedding(self, hidden_lstm, activation):
        """ LSTM Network """

        for _ in range(hidden_lstm):
            self.net.add(
                LSTM(32, input_shape=(self.max_sequence_len,), return_sequences=True)
            )
        self.net.add(LSTM(32, input_shape=(self.max_sequence_len,)))
        self.net.add(Dense(self.total_words, activation=activation))

        return self
