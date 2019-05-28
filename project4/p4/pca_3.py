import sys
import math
import pandas as pd
import numpy as np
from numpy.linalg import eig

df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)
dfm = df.values

def pca(data):
    M = np.mean(data.T, axis=1)
    C = data - M
    V = np.cov(C.T)
    eValues, eVectors = eig(V)
    P = eVectors.T.dot(C.T)
    return eValues[:10]

print(pca(dfm))
