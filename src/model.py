from itertools import islice

import keras.utils as ku
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator

from .metrics import perplexity_raw


def batchedgenerator(generatorfunction):
    """Decorator that makes a pattern generator produce patterns in batches
    A "batchsize" parameter is added to the generator, that if specified
    groups the data in batches of such size.
    
    It is expected that the generator returns instances of data patterns,
    as tuples of numpy arrays (X,y)
    """

    def modgenerator(*args, **kwargs):
        if "batchsize" in kwargs:
            batchsize = kwargs["batchsize"]
            del kwargs["batchsize"]
        else:
            batchsize = 1
        for batch in splitevery(generatorfunction(*args, **kwargs), batchsize):
            yield batch

    return modgenerator


def batchedpatternsgenerator(generatorfunction):
    """Decorator that assumes patterns (X,y) and stacks them in batches
    
    This can be thought of a specialized version of the batchedgenerator
    that assumes the base generator returns instances of data patterns,
    as tuples of numpy arrays (X,y). When grouping them in batches the
    numpy arrays are stacked so that each returned batch has a pattern 
    per row.
    
    A "batchsize" parameter is added to the generator, that if specified
    groups the data in batches of such size.
    """

    def modgenerator(*args, **kwargs):
        for batch in batchedgenerator(generatorfunction)(*args, **kwargs):
            Xb, yb = zip(*batch)
            yield np.stack(Xb), np.stack(yb)

    return modgenerator


def infinitegenerator(generatorfunction):
    """Decorator that makes a generator replay indefinitely
    
    An "infinite" parameter is added to the generator, that if set to True
    makes the generator loop indifenitely.    
    """

    def infgenerator(*args, **kwargs):
        if "infinite" in kwargs:
            infinite = kwargs["infinite"]
            del kwargs["infinite"]
        else:
            infinite = False
        if infinite == True:
            while True:
                for elem in generatorfunction(*args, **kwargs):
                    yield elem
        else:
            for elem in generatorfunction(*args, **kwargs):
                yield elem

    return infgenerator


class BaseNetwork(BaseEstimator):
    def __init__(self, tokenizer=Tokenizer(), max_sequence_len=301, batchsize=32):
        """ AÑADIR MAS INFO
        usamos el max_sequence_len porque así si la longitud máxima de una frase es descabellada
        nos cubrimos las espaldas"""
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.batchsize = batchsize

    def etl(self, data):

        # basic cleanup
        corpus = data.lower().split("\n")

        # tokenization
        self.tokenizer.fit_on_texts(corpus)
        self.total_words = len(self.tokenizer.word_index) + 1

        # Total samples
        self.num_train_samples = len(list(self.patterngenerator(corpus, batchsize=self.batchsize, count=True)))

        return corpus

    def patterngenerator(self, corpus, **kwargs):
        """Infinite generator of encoded patterns.
        
        Arguments
            - corpus: iterable of strings making up the corpus
            - **kwargs: any other arguments are passed on to decodetext
        """
        # Pre-tokenized all corpus documents, for efficiency
        tokenizedcorpus = self.tokenizer.texts_to_sequences(corpus)
        self.max_sequence_len = min(
            len(max(tokenizedcorpus, key=len)), self.max_sequence_len
        )
        for pattern in self._tokenizedpatterngenerator(tokenizedcorpus, **kwargs):
            yield pattern

    @infinitegenerator
    @batchedpatternsgenerator
    def _tokenizedpatterngenerator(self, tokenizedcorpus, **kwargs):
        for token_list in tokenizedcorpus:
            for i in range(1, len(token_list)):
                sample = np.array(
                    pad_sequences(
                        [token_list[: i + 1]],
                        maxlen=self.max_sequence_len,
                        padding="pre",
                        value=0,
                    )
                )
                X, y = sample[:, :-1], sample[:, -1]
                y = ku.to_categorical(y, num_classes=self.total_words)
                if "count" in kwargs and kwargs["count"] is True:
                    yield 0, 0
                else:
                    yield X[0], y[0]

    def generate_text(self, seed_text, next_words):

        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.max_sequence_len - 1, padding="pre"
            )
            predicted = self.net.predict(token_list, verbose=0)[0]
            sampled_predicted = sample(np.log(predicted), 0.5)
            seed_text += f" {self.tokenizer.index_word[sampled_predicted]}"

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
        corpus,
        earlystop=None,
        epochs=200,
        max_queue_size=1000, #qué debería poner?
        shuffle=False,
        use_multiprocessing=True,
        verbose=1,
    ):

        if earlystop is True:
            earlystop = [
                EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto"
                )
            ]
        print("The fit process is starting!")
        self.net.fit_generator(
            self.patterngenerator(corpus, batchsize=self.batchsize, infinite=True),
            steps_per_epoch=self.num_train_samples,
            callbacks=earlystop,
            epochs=epochs,
            # validation_data=self.patterngenerator(corpus, batchsize=batchsize, infinite=True),
            verbose=verbose,
            max_queue_size=max_queue_size,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
        )

    def Baseline(self):
        """Simple network with an embedding layer and a dense one"""

        self.net.add(Flatten())

        return self

    def LSTM_Embedding(self, gpu, hidden_lstm, lstm_units):
        """ LSTM Network """

        if gpu:
            from keras.layers import CuDNNLSTM
            lstm = CuDNNLSTM
        else:
            lstm = LSTM
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
        embedding_matrix = np.random.normal(
            emb_mean, emb_std, (self.total_words, embedding_size)
        )
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


def splitevery(iterable, n):
    """Returns blocks of elements from an iterator"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))
