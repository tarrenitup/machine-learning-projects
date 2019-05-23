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

def makeCentroids(k): # make k vectors, each w/ 
    arr = np.zeros(k)
    for i in range(arr.size):
        arr[i] = getRand(255)
    return arr


### get data ###
dfSample = pd.read_csv('test-data.txt', dtype='int', delimiter = ',', header=None)
# dfFirst2Rows = pd.read_csv('p4-data.txt', header=None, nrows=2)
# df = pd.read_csv('p4-data.txt', header=None)


### get parameter(s) ###
k = 3 # default
if len(sys.argv) > 1:
    k = int(sys.argv[1]) # if user supplies parameter


### initialization ###

# make a k long list of random centroids (values between 0 and 255)
sampleCentroids = makeCentroids(3)
centroids = makeCentroids(k)




# print(  )
