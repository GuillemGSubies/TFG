# @author Guillem G. Subies

import numpy as np


def sample(logprobs, temperature=1.0):
    """Modifies probabilities with a given temperature, to add creativity

    Parameters
    ----------
    logprobs : list of float
        List of log probabilities
    temperature : float, optional
        How much it will be modified. More modification if lowe.

    Returns
    -------
        The index of the chosen element
    """
    probs = np.exp(logprobs / temperature)
    normprobs = normalize(probs)
    return np.argmax(np.random.multinomial(1, normprobs, 1))


def normalize(probs):
    """Normalizes a list of probabilities, so that they sum up to 1"""
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]
