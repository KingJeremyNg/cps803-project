import numpy as np
from compare_results import compareResults
from sklearn.neighbors import KNeighborsClassifier


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier().fit(trainX, trainY)
    prediction = knn.predict(testX)
    prediction_train = knn.predict(trainX)
    compareResults(prediction_train, trainY, note=note + " train")
    compareResults(prediction, testY, note=note + " valid")
