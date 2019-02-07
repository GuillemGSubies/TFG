# @author Guillem G. Subies

import keras.utils as ku
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, CuDNNLSTM, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator

from .generators import batchedpatternsgenerator, infinitegenerator, maskedgenerator
from .metrics import perplexity_raw
from .plotting import plot_history as _plot_history
from .utils import sample


class BaseNetwork(BaseEstimator):
    """Class to built, train and generate new text with neural networks"""

    ###############################################################################
    ##################################Main methods#################################
    ###############################################################################

    def __init__(
        self, tokenizer=None, max_sequence_len=301, vocab_size=None, batchsize=32
    ):
        """
        Parameters
        ----------
        tokenizer : object, optional
            If None, a default tokanizer will be used. 
        max_sequence_len : int, optional
            Maximum lenght, in words, of each sample. It will be the minimum between
            the introduced number and the maximum lenght of the saples befor being processed
        vocab_size : int, optional
            If None, it will be the same as the full vocabulary. Else, the maximum
            size of the vocabulary will be vocab_size.
        batchsize: int, optional
            Size of batches to pass to the fit_generator

        """

        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(oov_token=None) if tokenizer is None else tokenizer
        self.max_sequence_len = max_sequence_len
        self.batchsize = batchsize

    def etl(self, data, mask=[True, True, True, False]):
        """Method that preprocesses input and creates some necesary variables to be used
        in the fit_generator
        
        Parameters
        ----------
        data : iterable of strings
            Dataset we want to use
        mask : list of bool, optional
            Mask to be used in the train test split. For instance the default value means
            that every fourth sample generated will be used in the validation step. If mask
            is None, no train test split will be done.

        Returns
        -------
        corpus : iterable of strings
            Preprocessed dataset to be used in the fit phase.
        """

        # Basic cleanup
        corpus = data.lower().split("\n")

        # Tokenization
        self.tokenizer.fit_on_texts(corpus)
        if self.vocab_size is not None and self.vocab_size < len(
            self.tokenizer.word_index
        ):
            sorted_dict = [
                (key, self.tokenizer.word_index[key])
                for key in sorted(
                    self.tokenizer.word_counts,
                    key=self.tokenizer.word_counts.get,
                    reverse=True,
                )
            ][: self.vocab_size - 1]

            self.tokenizer.word_index = dict(sorted_dict)
            self.tokenizer.index_word = dict(
                zip(
                    list(self.tokenizer.word_index.values()),
                    list(self.tokenizer.word_index.keys()),
                )
            )
            assert (
                self.vocab_size - 1
                == len(self.tokenizer.word_index)
                == len(self.tokenizer.index_word)
            )
        else:
            self.vocab_size = len(self.tokenizer.word_index) + 1

        # Prepare masks
        if mask is not None:
            self.testmask = [not x for x in mask]
            self.mask = mask
        else:
            self.mask, self.testmask = [True], [True]

        # Total samples
        self.num_train_samples = len(
            list(
                self.patterngenerator(
                    corpus, batchsize=self.batchsize, count=True, mask=self.mask
                )
            )
        )
        self.num_test_samples = len(
            list(
                self.patterngenerator(
                    corpus, batchsize=self.batchsize, count=True, mask=self.testmask
                )
            )
        )

        print(f"There are a total of {self.num_train_samples} training samples")
        print(f"There are a total of {self.num_test_samples} validation samples")

        return corpus

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
        lstm=[32],
    ):
        """Builds the architecture of a neural network

        Parameters
        ----------
        activation : str, optional
            Activation function used in the las layer of the network.
        arch : str, optional
            If "Baseline": The architecture will be embedding layer + dense.
            If "LSTM_Embedding": The architectura will be embedding layer + bidirectional
                LSTM + hidden LSTMs + dense. The hidden LSTMs will be defined by the param
                "lstm".
        embedding : str, optional
            If None, a simple embedding layer will be used. If "fastText", fastText
            embeddings will be used.
        embedding_output_dim : int, optional
            Outpud dimension of the embedding layer. It is ignored if the used embedding
            is "fastText" (it has a fixed size of 300)
        gpu : bool, optional
            If True, CuDNNLSTM networks will be used in stead of LSTM.
        """

        self.net = Sequential()

        # Embedding layer
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
                input_dim=self.vocab_size,
                output_dim=output_dim,
                input_length=self.max_sequence_len - 1,
                trainable=trainable,
                weights=weights,
            )
        )

        if arch == "Baseline":
            self.net.add(Flatten())
        elif arch == "LSTM_Embedding":
            self.LSTM_Embedding(gpu=gpu, lstm_arch=lstm)
        else:
            raise Exception("Unknown network architecture")

        self.net.add(Dense(self.vocab_size, activation=activation))
        self.net.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        print(self.summary())

    def fit(
        self,
        corpus,
        earlystop=None,
        epochs=200,
        max_queue_size=1000,  # qué debería poner?
        save=False,
        shuffle=False,
        use_multiprocessing=True,
        verbose=1,
        plot=True,
    ):

        """Fits the model with given data. It uses generators to create train an test samples

        Parameters
        ----------
        corpus : list of str
            Dataset to train the model.
        earlystop : bool or callable, optional
            If False no callbacks will be used. If True, a simple EarlyStopping will be used.
            A custom callback method can also be passed to the fit method.
        epochs : int, optional
            Number of train epochs. 
        max_queue_size : int, optional
            Like in keras
        save : str or bool
            Whether to save or not the model in a file. If False it will not be saved.
            If strm it will be saved in path=str.
        shuffle : bool, optional
            Wether so shufle or not train samples during the training process.
        use_multiprocessing : bool, optional
            Like in keras.
        verbose : int, optional
            Like in keras.
        plot : bool, optional
            Wether to plot the training history at the end of the training or not.
        Returns
        -----
        data : dict
            Embedding dict.

        """

        if earlystop is True:
            earlystop = [
                EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto"
                )
            ]

        print("The fit process is starting!")
        self.net.fit_generator(
            self.patterngenerator(
                corpus, batchsize=self.batchsize, infinite=True, mask=self.mask
            ),
            steps_per_epoch=self.num_train_samples,
            callbacks=earlystop,
            epochs=epochs,
            validation_data=self.patterngenerator(
                corpus, batchsize=self.batchsize, infinite=True, mask=self.testmask
            ),
            validation_steps=self.num_test_samples,
            verbose=verbose,
            max_queue_size=max_queue_size,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
        )

        if save:
            self.save(save)

        if plot:
            self.plot_history() 

    def generate_text(self, seed_text, next_words):

        for i in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.max_sequence_len - 1, padding="pre"
            )
            predicted = self.net.predict(token_list, verbose=0)[0]
            sampled_predicted = sample(np.log(predicted), 0.5)
            print(f"why?{i}")
            print(sampled_predicted)
            seed_text += f" {self.tokenizer.index_word[sampled_predicted]}"

        return seed_text

    ###############################################################################
    ##################################Aux methods##################################
    ###############################################################################

    def LSTM_Embedding(self, gpu, lstm_arch):
        """This Network consists in a embedding layer followed by a bidirectional
        LSTM and some number of hidden LSTM. There is a dense layer at the end.


        Parameters
        ----------
        gpu : bool
            If True, CuDNNLSTM networks will be used in stead of LSTM
        lstm_arch : list of int
            len(lstm_arch) will be the number of LSTM layers in the model (being the
            first one, bidirectional) and every elem in lstm_arch is the number of
            neurons for the ith layer.

        Returns
        -----
        self
        """

        lstm = CuDNNLSTM if gpu else LSTM

        # Bidirectional layer
        layer = lstm(
            lstm_arch.pop(0),
            input_shape=(self.max_sequence_len,),
            return_sequences=False if len(lstm_arch) == 0 else True,
        )
        self.net.add(Bidirectional(layer, merge_mode="concat"))

        # Hidden layers
        for i, elem in enumerate(lstm_arch):
            self.net.add(
                lstm(
                    elem,
                    input_shape=(self.max_sequence_len,),
                    return_sequences=True if i < len(lstm_arch) - 1 else False,
                )
            )

        return self

    def summary(self):
        """Wrapper method for keras' sequential model summary"""
        return self.net.summary()

    @property
    def history(self):
        """keras' sequential model history"""
        return self.net.history

    def plot_history(self):
        """Wrapper method for plotting the model history"""
        _plot_history(self.history)

    def save(self, path):
        """Wrapper method for keras' sequential model save"""
        return self.net.save(path)


    ###############################################################################
    ###########################Embedding related methods###########################
    ###############################################################################

    def load_vectors_words(self, fname):
        """Loads embeddings from a FastText file. Only loads embeddings for the given
        dictionary of words

        Parameters
        ----------
        fname : str
            Location of the embbeding file

        Returns
        -----
        data : dict
            Embedding dict

        """

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
        """Creates a weight matrix for an Embedding layer using an embeddings dictionary

        Parameters
        ----------
        embeddings : dict
            preloaded embedding dict to use

        Returns
        -------
        embedding_matrix : numpy.ndarray
            Matrix with"""

        # Compute mean and standard deviation for embeddings
        all_embs = np.stack(embeddings.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_size = len(next(iter(embeddings.values())))
        embedding_matrix = np.random.normal(
            emb_mean, emb_std, (self.vocab_size, embedding_size)
        )
        for word, i in self.tokenizer.word_index.items():
            if i >= self.vocab_size:
                break
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    ###############################################################################
    #########################fit_generator related methods#########################
    ###############################################################################

    def patterngenerator(self, corpus, **kwargs):
        """Infinite generator of encoded patterns.
        
        Parameters
        -----------
            corpus : iterable of strings
                The corpus
            **kwargs : any other arguments are passed on to decodetext
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
    @maskedgenerator
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
                y = ku.to_categorical(y, num_classes=self.vocab_size)
                if "count" in kwargs and kwargs["count"] is True:
                    yield 0, 0
                else:
                    yield X[0], y[0]


def load_model(path):
    """PROVISIONAL Wrapper method for keras' sequential model load_model"""
    return load_model(path)

    # @classmethod
    # def load_model(cls, path):
    # TODO: Esto es más complicado de lo que parece ya que habría que guardar también el tokenizador en algún sitio también
    #     """Wrapper method for keras' load_model"""

    #     model = cls()
    #     return load_model(path)
