import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier().fit(trainX, trainY)
    prediction = knn.predict(testX)
    prediction_train = knn.predict(trainX)
    compareResults(prediction_train, trainY, note=note + " train")
    compareResults(prediction, testY, note=note + " valid")
    #bias_variance(trainX, trainY, testX, testY, knn, note)
    print(f'CV score: {cross_val_score(knn, testX, testY, cv=3).mean()}')
