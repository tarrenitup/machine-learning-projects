import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd

# loading training data
knnDummyData = pd.read_csv('data/knn_dummy.csv', header=None)
knnTrainData = pd.read_csv('data/knn_train.csv', header=None)
knnTestData = pd.read_csv('data/knn_test.csv', header=None)

# Separate into labels and features

knnDummyLabels = knnDummyData.iloc[:, :1].values
knnDummyFeatures = knnDummyData.iloc[:, 1:].values 

print(knnDummyFeatures)

# knnTrainLabels = 
# knnTrainFeatures = 

# knnTestLabels = 
# knnTestFeatures = 