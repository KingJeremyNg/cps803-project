import numpy as np
from compare_results import compareResults
from sklearn.neighbors import KNeighborsClassifier


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier()
    knn.fit(trainX, trainY)
    prediction = knn.predict(testX)
    prediction_train = knn.predict(trainX)

    pred_prob = knn.predict_proba(testX)
    pred_prob_train = knn.predict_proba(trainX)
    labels = knn.classes_

    compareResults(prediction_train, trainY, pred_prob_train, labels, note=note + " train")
    compareResults(prediction, testY, pred_prob, labels, note=note + " valid")
