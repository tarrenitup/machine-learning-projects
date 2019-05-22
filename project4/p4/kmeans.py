import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#initialization
tdf = pd.read_csv('p4-data.txt', header=None, nrows=2)
df = pd.read_csv('p4-data.txt', header=None)

np.random.seed(200)

k = 3
if len(sys.argv) > 1:
    k = float(sys.argv[1])

print(tdf)



# centroids = {

# }