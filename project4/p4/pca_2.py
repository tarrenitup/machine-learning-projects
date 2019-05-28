import sys
import math
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

dfS = pd.read_csv('S4-data.txt', dtype='int', delimiter = ',', header=None)
dfSm = dfS.values
dfF = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
dfFm = dfF.values
df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)
dfm = df.values

def pca(data):
    M = np.mean(data.T, axis=1)

    print("Means: ", M) 
    # plt.plot(M, 'ro') # uncomment to view plot
    # plt.show()

    C = data - M
    V = np.cov(C.T)
    eValues, eVectors = eig(V)

    print("Eigenvectors: ", eVectors) 
    # plt.plot(eVectors[:10], 'ro') # uncomment to view plot
    # plt.show() # uncomment to view plot

    P = eVectors.T.dot(C.T)
    return eValues[:10]

pca(dfm)
