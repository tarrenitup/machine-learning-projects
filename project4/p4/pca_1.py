import sys
import math
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

### get data ###

dfS = pd.read_csv('S4-data.txt', dtype='int', delimiter = ',', header=None)
dfSm = dfS.values
dfF = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
dfFm = dfF.values
df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)
dfm = df.values

def pca(data):

    ### get the mean of each column ###
    # in: dim x examples matrix of data, out: col size vector of col means
    M = np.mean(data.T, axis=1)
    # print("means: ", M)
    # print("####################")


    ### get center values ### 
    # subtracting values from the mean for each col making a new set of data #
    C = data - M
    # print("center cols: ", C)
    # print("####################")


    ### calculate the covarience matrix ###
    # correlation (normalized): the amount and direction (+/-) that two vectors change together (from the mean?..)
    # covariance (unnormalized, generalized) version of correlation across multiple columns.
    # covariance matrix: stores covarience scores for every col with everyother col
    V = np.cov(C.T)
    # print("covarience matrix: ", V)
    # print("####################")


    ### calculate eigendecomposition ###
    # result: list of eignvalues and a list of eignvectors #
    # eignvectors sorted by eignvalues in desc order for _ #
    # evalues close to 0 may be discarded,
    eValues, eVectors = eig(V)
    # print("eigenvectors: ", eVectors)
    # print("eigenvalues: ", eValues)
    # print("####################")


    ### project data ###
    P = eVectors.T.dot(C.T)
    # print("projected data: ", P.T)
    # print("####################")

    return eValues[:10]

print(pca(dfm))
