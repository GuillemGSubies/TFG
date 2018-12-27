from keras import backend as K


# source https://stackoverflow.com/questions/44697318/how-to-implement-perplexity-in-keras
def perplexity_raw(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    cross_entropy = K.cast(
        K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
        K.floatx(),
    )
    perplexity = K.exp(cross_entropy)
    return perplexity
