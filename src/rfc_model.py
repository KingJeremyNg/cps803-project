import numpy as np
from compare_results import compareResults
from sklearn.ensemble import RandomForestClassifier


def rfcModel(trainX, trainY, testX, testY, note="Unknown"):
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(trainX, trainY)
    pred = rfc.predict(testX)
    pred_train = rfc.predict(trainX)

    pred_prob = rfc.predict_proba(testX)
    pred_prob_train = rfc.predict_proba(trainX)
    labels = rfc.classes_

    compareResults(pred_train, trainY, pred_prob_train, labels, note=note + " train")
    compareResults(pred, testY, pred_prob, labels, note=note + " valid")
