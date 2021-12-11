import numpy as np
from compare_results import compareResults
from sklearn import svm


def svmModel(trainX, trainY, testX, testY, note="Unknown"):
    model = svm.SVC(probability=True)
    trained = model.fit(trainX, trainY)

    prediction = trained.predict(testX)
    prediction_train = trained.predict(trainX)

    pred_prob = model.predict_proba(testX)
    pred_prob_train = model.predict_proba(trainX)
    labels = model.classes_

    compareResults(prediction_train, trainY, pred_prob_train, labels, note=note + " train")
    compareResults(prediction, testY, pred_prob, labels, note=note + " valid")
