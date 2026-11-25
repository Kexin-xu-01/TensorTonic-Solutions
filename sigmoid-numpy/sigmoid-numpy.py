import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """

    answer = 1/(np.exp(np.negative(x)) + 1)

    return answer
