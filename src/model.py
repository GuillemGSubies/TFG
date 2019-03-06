# @author Guillem G. Subies


import datetime
import json
import zipfile
from math import ceil
from subprocess import check_call

import jsonpickle
import keras.utils as ku
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    LSTM,
    Bidirectional,
    CuDNNLSTM,
    Dense,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
)
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator

from .callbacks import ModelFullCheckpoint
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
        self,
        tokenizer=None,
        max_sequence_len=301,
        min_word_appearences=None,
        vocab_size=None,
        batchsize=32,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tokenizer : object, optional
            If None, a default tokanizer will be used.
        max_sequence_len : int, optional
            Maximum lenght, in words, of each sample. It will be the minimum between
            the introduced number and the maximum lenght of the saples befor being processed
        min_word_appearences : int, optional
            Minimum number of appearences of a word in the text in order to take it into account
            This must not be used at the same time that vocab_size
        vocab_size : int, optional
            If None, it will be the same as the full vocabulary. Else, the maximum
            size of the vocabulary will be vocab_size.
        batchsize: int, optional
            Size of batches to pass to the fit_generator

        """

        self.vocab_size = vocab_size
        self.min_word_appearences = min_word_appearences
        if self.vocab_size and self.min_word_appearences:
            raise AttributeException(
                "You must specify only vocab_size or min_word_appearences, not both."
            )
        self.tokenizer = (
            Tokenizer(num_words=vocab_size, oov_token=None)
            if tokenizer is None
            else tokenizer
        )
        self.max_sequence_len = max_sequence_len
        self.batchsize = batchsize
        # Other kwargs, this is used in the load_model method
        self.__dict__.update(kwargs)

    def etl(self, data, mask=None):
        """Method that preprocesses input and creates some necesary variables to be used
        in the fit_generator.

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

        mask = mask or [True, True, True, False]
        # Basic cleanup
        corpus = data.lower().split("\n")

        # Tokenization
        self.tokenizer.fit_on_texts(corpus)
        if self.min_word_appearences:
            low_count_words = [
                word
                for word, count in self.tokenizer.word_counts.items()
                if count < self.min_word_appearences
            ]
            for word in low_count_words:
                del self.tokenizer.word_index[word]
                del self.tokenizer.word_docs[word]
                del self.tokenizer.word_counts[word]
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
        arch="MaxPooling",
        embedding=None,
        embedding_output_dim=64,
        gpu=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer="adam",
        lstm=None,
        **kwargs,
    ):
        """Builds the architecture of a neural network

        Parameters
        ----------
        activation : str, optional
            Activation function used in the las layer of the network.
        arch : str, optional
            If "Baseline": The architecture will be embedding layer + flatten + dense. This is VERY memory hungry
            If "MaxPooling": The architecture will be embedding layer + GlobalMaxPooling1D + dense. This is VERY memory hungry
            If "AveragePooling": The architecture will be embedding layer + GlobalAveragePooling1D + dense. This is VERY memory hungry
            If "LSTM_Embedding": The architectura will be embedding layer + bidirectional
                LSTM + hidden LSTMs + dense. The hidden LSTMs will be defined by the param
                "lstm".
        embedding : str, optional
            If None, a simple embedding layer will be used. If "fastText", fastText
            embeddings will be used. fastText embeddings must be downloaded and uncompressed
            in the project root
        embedding_output_dim : int, optional
            Outpud dimension of the embedding layer. It is ignored if the used embedding
            is "fastText" (it has a fixed size of 300)
        gpu : bool, optional
            If True, CuDNNLSTM networks will be used in stead of LSTM.
        """

        lstm = lstm or [32]
        metrics = metrics or [perplexity_raw]
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
                    "No embedding file found, downloading it... (this will take a while)"
                )
                check_call(
                    f"curl -L# 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/{fastText_file}.zip'",
                    shell=True,
                )
                with zipfile.ZipFile(f"{fastText_file}.zip", "r") as file:
                    file.extractall("./")
                embeddings = self.load_vectors_words(fastText_file)
                print("Embedding file loaded sucessfully!")
            finally:
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

        # Core layers
        if arch == "Baseline":
            self.net.add(Flatten())
        elif arch == "MaxPooling":
            self.net.add(GlobalMaxPooling1D())
        elif arch == "AveragePooling":
            self.net.add(GlobalAveragePooling1D())
        elif arch == "LSTM_Embedding":
            self.LSTM_Embedding(gpu=gpu, lstm_arch=lstm)
        else:
            raise Exception("Unknown network architecture")

        # Final layer
        self.net.add(Dense(self.vocab_size, activation=activation))
        self.net.compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

        print(self.summary())

    def fit(
        self,
        corpus,
        callbacks=None,
        checkpoints=True,
        dynamic_lr=True,
        earlystop=True,
        epochs=200,
        verbose=1,
        plot=True,
        **kwargs,
    ):

        """Fits the model with given data. It uses generators to create train an test samples

        Parameters
        ----------
        corpus : list of str
            Dataset to train the model.
        callbacks : object, optional
        checkpoints : bool, optional
            If True, ModelFullCheckpoint will be added to callbacks
        dynamic_lr : bool, optional
            If True, ModelFullCheckpoint will be added to callbacks
        earlystop : bool, optional
            If False no default earlystop will be used. If True, a simple EarlyStopping will be used.
        epochs : int, optional
            Number of train epochs.
        save : str or bool
            Whether to save or not the model in a file. If False it will not be saved.
            If strm it will be saved in path=str.
        verbose : int, optional
            Like in keras.
        plot : bool, optional
            Wether to plot the training history at the end of the training or not.
        Returns
        -----
        data : dict
            Embedding dict.

        """

        callbacks = callbacks or []
        if dynamic_lr:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.8, patience=5, verbose=1, mode="min"
                )
            )
        if earlystop:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="min"
                )
            )
        if checkpoints:
            model_name = f"Best_model_{datetime.datetime.now().time()}"
            print(f"The model will be saved with the name: {model_name}")
            callbacks.append(
                ModelFullCheckpoint(
                    modelo=self,
                    filepath=model_name,
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                )
            )
        print("The fit process is starting!")
        self.net.fit_generator(
            self.patterngenerator(
                corpus, batchsize=self.batchsize, infinite=True, mask=self.mask
            ),
            steps_per_epoch=ceil(self.num_train_samples / self.batchsize),
            callbacks=callbacks,
            epochs=epochs,
            validation_data=self.patterngenerator(
                corpus, batchsize=self.batchsize, infinite=True, mask=self.testmask
            ),
            validation_steps=ceil(self.num_test_samples / self.batchsize),
            verbose=verbose,
            **kwargs,
        )
        if plot:
            self.plot_history()

    def generate_text(self, seed_text, next_words):
        """Generates text following the given seed

        Parameters
        ----------
        seed_text : str
            String to start generating text from (what you pass to the predict method).
        next_words : int
            Number of words to generate

        Returns
        -------
        generated_text : str
            String containing the generated text
        """

        generated_text = seed_text
        for i in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([generated_text])
            token_list = pad_sequences(
                token_list, maxlen=self.max_sequence_len - 1, padding="pre"
            )
            predicted = self.net.predict(token_list, verbose=0)[0]
            sampled_predicted = sample(np.log(predicted), 0.5)
            try:
                generated_text += (
                    f" {self.tokenizer.sequences_to_texts([[sampled_predicted]])[0]}"
                )
            except:
                # Predicted 0, pass this time
                pass

        return generated_text

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

    def save(self, path=None):
        """Saves the model in json format. The keras network will be
        saved into a file called path_network.h5 and the rest of
        the params into path_attrs.json"""
        if path is None:
            path = f"{self}"
        kwargs = dict()
        for key in self.__dict__:
            if key != "net":
                kwargs[key] = self.__dict__[key]
        try:
            self.net.save(f"{path}_network.h5")
            with open(f"{path}_attrs.json", "w") as outfile:
                json.dump(jsonpickle.encode(kwargs), outfile)
            return "Model saved successfully!"
        except:
            return "Something went wrong when saving the model..."

    @classmethod
    def load_model(cls, path):
        """Loads a model from the files. The keras network should be in a file called
        path_network.h5 and the rest of the params in path_attrs.json"""
        with open(f"{path}_attrs.json") as infile:
            kwargs = jsonpickle.decode(json.load(infile))
        kwargs["net"] = load_model(
            f"{path}_network.h5", custom_objects={"perplexity_raw": perplexity_raw}
        )
        return cls(**kwargs)

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
        vocab = tuple(self.tokenizer.word_index.keys())[: self.vocab_size - 1]
        with open(fname) as fin:
            next(fin)  # Skip first line, just contains embeddings size data
            for line in fin:
                tokens = line.rstrip().split(" ")
                word = tokens[0]
                if word in vocab:
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
        vocab = tuple(self.tokenizer.word_index.items())[: self.vocab_size - 1]
        for word, i in vocab:
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    ###############################################################################
    #########################fit_generator related methods#########################
    ####################Inspired by Alvaro Barbero's neurowriter###################
    ###https://github.com/albarji/neurowriter/blob/master/neurowriter/encoding.py##
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
                sampl = np.array(
                    pad_sequences(
                        [token_list[: i + 1]],
                        maxlen=self.max_sequence_len,
                        padding="pre",
                        value=0,
                    )
                )
                X, y = sampl[:, :-1], sampl[:, -1]
                y = ku.to_categorical(y, num_classes=self.vocab_size)
                if "count" in kwargs and kwargs["count"] is True:
                    yield 0, 0
                else:
                    yield X[0], y[0]
