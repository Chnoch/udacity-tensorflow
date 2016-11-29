
scores = [2.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

print(softmax(np.multiply(scores, 10)))


