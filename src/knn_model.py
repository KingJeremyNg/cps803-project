import numpy as np
from compare_results import compareResults
from sklearn.neighbors import KNeighborsClassifier


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier().fit(trainX, trainY)
    prediction = knn.predict(testX)
    compareResults(prediction, testY, note=note)
