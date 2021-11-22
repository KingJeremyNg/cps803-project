import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def readDataset(path):
    X = pd.read_csv(path, usecols=[i for i in range(11)]).to_numpy()
    y = pd.read_csv(path, usecols=["quality"]).to_numpy()
    trainX, testX, trainY, testY = train_test_split(
        X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.fit_transform(testX)
    # train_test_split is giving dim(n, 1) when it should be dim(n,)
    return trainX, trainY[:, 0], testX, testY[:, 0]
