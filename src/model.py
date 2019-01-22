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
    def __init__(self, tokenizer=Tokenizer(), max_sequence_len=301):
        """ AÑADIR MAS INFO
        usamos el max_sequence_len porque así si la longitud máxima de una frase es descabellada
        nos cubrimos las espaldas"""
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len

    def etl(self, data):

        # basic cleanup
        corpus = data.lower().split("\n")

        # tokenization
        self.tokenizer.fit_on_texts(corpus)
        self.total_words = len(self.tokenizer.word_index) + 1

        # create input sequences using list of tokens
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                input_sequences.append(token_list[: i + 1])

        # pad sequences
        self.max_sequence_len = min(
            len(max(input_sequences, key=len)), self.max_sequence_len
        )
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
            sampled_predicted = sample(np.log(predicted), 0.5)
            seed_text += f" {self.tokenizer.index_word[sampled_predicted]}"

        return seed_text

    def compile(
        self,
        activation="softmax",
        arch="Baseline",
        embedding=None,
        embedding_output_dim=64,
        gpu=False,
        loss="categorical_crossentropy",
        metrics=[perplexity_raw],
        optimizer="adam",
        hidden_lstm=0,
        lstm_units=32,
    ):

        self.net = Sequential()

        # Embeddign layer
        output_dim = embedding_output_dim
        trainable = True
        weights = None
        if embedding == "fastText":
            fastText_file = "crawl-300d-2M.vec"
            try:
                embeddings = self.load_vectors_words(fastText_file)
                print("Embedding file loaded sucessfully!")
            except:
                print(
                    "Are you sure that you downloaded the embeddings? The model will work without fastText"
                )
            else:
                embedding_matrix = self.create_embedding_matrix(embeddings)
                output_dim = embedding_matrix.shape[1]
                trainable = False
                weights = [embedding_matrix]

        self.net.add(
            Embedding(
                input_dim=self.total_words,
                output_dim=output_dim,
                input_length=self.max_sequence_len - 1,
                trainable=trainable,
                weights=weights,
            )
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
        batch_size=32,
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
        print("The fit process is starting!")
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

    def load_vectors_words(self, fname):
        """Loads embeddings from a FastText file. Only loads embeddings for the given dictionary of words"""
        data = {}
        with open(fname) as fin:
            next(fin)  # Skip first line, just contains embeddings size data
            for line in fin:
                tokens = line.rstrip().split(" ")
                word = tokens[0]
                if word in self.tokenizer.word_index:
                    data[word] = np.array(list(map(float, tokens[1:])))
        return data

    def create_embedding_matrix(self, embeddings):
        """Creates a weight matrix for an Embedding layer using an embeddings dictionary and a Tokenizer"""

        # Compute mean and standard deviation for embeddings
        all_embs = np.stack(embeddings.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_size = len(next(iter(embeddings.values())))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (self.total_words, embedding_size))
        for word, i in self.tokenizer.word_index.items():
            if i >= self.total_words:
                break
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix



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
