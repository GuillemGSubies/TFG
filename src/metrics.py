# @author Guillem G. Subies

from keras import backend as K
from keras.losses import categorical_crossentropy


def perplexity(y_true, y_pred):
    crossentropymean = K.mean(K.categorical_crossentropy(y_true, y_pred))
    return K.pow(2.0, crossentropymean)


def perplexity_e(y_true, y_pred):
    crossentropymean = K.mean(K.categorical_crossentropy(y_true, y_pred))
    return K.exp(crossentropymean)


def perplexity_vocab(vocab_size, expo="e"):
    #WIP
    def ppl(y_true, y_pred):
        crossentropysumdiv = K.sum(K.categorical_crossentropy(y_true, y_pred))/vocab_size
        if expo == "e":
            return K.exp(crossentropysumdiv)
        elif isinstance(int, expo):
            # Usually it will be 2
            return K.pow(expo, crossentropysumdiv)
    return ppl
