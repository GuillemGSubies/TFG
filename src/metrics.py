import keras.backend as K

def perplexity(y_true, y_pred):
    """TODO: AÑADIR DESCRIPCION"""
    return K.pow(2.0, K.mean(K.nn.softmax_cross_entropy_with_logits(y_true, y_pred, name=None)))