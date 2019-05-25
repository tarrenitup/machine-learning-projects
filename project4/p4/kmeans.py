import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### utilities ###
np.random.seed(200) # maybe make this based on time or something ..

def numRows(df):
    return len(df)

def numCols(df):
    return len(df.columns)

def getRand(cap):
    return np.random.randint(0,cap)

def dfRowToVector(df, rowIdx):
    vector = np.zeros(numCols(df))
    for i in range(vector.size):
        vector[i] = df.iloc[rowIdx][i]
    return vector



### get data ###
dfSample = pd.read_csv('test-data.txt', dtype='int', delimiter = ',', header=None)
dfFirst2Rows = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
# df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)


### get parameter(s) ###
k = 3 # default
if len(sys.argv) > 1:
    k = int(sys.argv[1]) # if user supplies parameter


### main ###

# pick k seeds
# def getKSeeds(k, df):
#     seeds = {}
#     for i in range(k):
#         randIdx = getRand(numRows(df))
#         seeds[i] = df[randIdx]
#     return seeds



# assign all points to its nearest seed

# compute k centroids

# reassign all points to its nearest centroid

# def assignment(df, centroids):
#     for k in centroids.keys(): # k times
#         df


# print(getKSeeds(1, dfSample))
# print( dfRowToVector(dfSample, 0) )
# print(dfSample.iloc[0])