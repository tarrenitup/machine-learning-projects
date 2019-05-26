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

def makeRandomVector(length, max):
    vector = np.zeros(length)
    for v in range(vector.size):
        vector[v] = getRand(max)
    return vector

def makeCentroids(k, df): # make k vectors, each w/ col random values between 0 and 255
    centroids = {}
    for i in range(k):
        centroids[i] = makeRandomVector(numCols(df), 255) # columns are dimensions
    return centroids

### get data ###
dfSample = pd.read_csv('test-data.txt', dtype='int', delimiter = ',', header=None)
dfFirst2Rows = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)


### get parameter(s) ###
k = 3 # default
if len(sys.argv) > 1:
    k = int(sys.argv[1]) # if user supplies parameter


### initialization ###
sampleCentroids = makeCentroids(3, dfSample)
# centroids = makeCentroids(k, df)


### assignment ###
def assignment(df, centroids):
    for k in centroids.keys(): # k times
        df

# print(sampleCentroids.keys())