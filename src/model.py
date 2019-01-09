import keras.utils as ku
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.layers import CuDNNLSTM, Bidirectional, Dense, Flatten, LSTM
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
            predicted = self.net.predict(token_list, verbose=0)[0]
            predicted = sample(np.log(predicted), 0.5)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text

    def compile(
        self,
        activation="softmax",
        arch="Baseline",
        gpu=False,
        loss="categorical_crossentropy",
        metrics=[perplexity_raw],
        optimizer="adam",
        hidden_lstm=0,
        lstm_units=32,
    ):

        self.net = Sequential()
        self.net.add(
            Embedding(self.total_words, 64, input_length=self.max_sequence_len - 1)
        )

        if arch == "Baseline":
            self.Baseline()
        elif arch == "LSTM_Embedding":
            self.LSTM_Embedding(gpu=gpu, hidden_lstm=hidden_lstm, lstm_units=lstm_units)
        else:
            raise Exception("Unknown network architecture")

        self.net.add(Dense(self.total_words, activation=activation))
        self.net.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print(self.summary())

    def fit(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        batch_size=16,
        earlystop=None,
        epochs=200,
        verbose=1,
    ):

        if earlystop is True:
            earlystop = [
                EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto"
                )
            ]
        self.net.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            callbacks=earlystop,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=verbose,
        )

    def Baseline(self):
        """Simple network with an embedding layer and a dense one"""

        self.net.add(Flatten())

        return self

    def LSTM_Embedding(self, gpu, hidden_lstm, lstm_units):
        """ LSTM Network """

        lstm = CuDNNLSTM if gpu else LSTM
        return_sequences = False if hidden_lstm == 0 else True

        layer = lstm(
            lstm_units,
            input_shape=(self.max_sequence_len,),
            return_sequences=return_sequences,
        )
        self.net.add(Bidirectional(layer, merge_mode="concat"))
        for _ in range(hidden_lstm - 1):
            self.net.add(
                lstm(
                    lstm_units,
                    input_shape=(self.max_sequence_len,),
                    return_sequences=True,
                )
            )
        if hidden_lstm >= 1:
            self.net.add(lstm(lstm_units, input_shape=(self.max_sequence_len,)))

        return self

    def summary(self):
        return self.net.summary()


def sample(logprobs, temperature=1.0):
    """Modifies probabilities with a given temperature, to add creativity
    Devuelve el indice"""
    probs = np.exp(logprobs / temperature)
    normprobs = normalize(probs)
    return np.argmax(np.random.multinomial(1, normprobs, 1))


def normalize(probs):
    """Normalizes a list of probabilities, so that they sum up to 1"""
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]
