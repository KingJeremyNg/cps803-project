import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier()
    knn.fit(trainX, trainY)
    prediction = knn.predict(testX)
    prediction_train = knn.predict(trainX)

    pred_prob = knn.predict_proba(testX)
    pred_prob_train = knn.predict_proba(trainX)
    labels = knn.classes_

    compareResults(prediction_train, trainY, pred_prob_train,
                   labels, note=note + " train")
    compareResults(prediction, testY, pred_prob, labels, note=note + " valid")

    print(f'CV score: {cross_val_score(knn, testX, testY, cv=3).mean()}')
