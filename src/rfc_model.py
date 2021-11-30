import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.ensemble import RandomForestClassifier


def rfcModel(trainX, trainY, testX, testY, note="Unknown"):
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(trainX, trainY)
    pred = rfc.predict(testX)
    pred_train = rfc.predict(trainX)
    compareResults(pred_train, trainY, note=note + " train")
    compareResults(pred, testY, note=note + " valid")
    #bias_variance(trainX, trainY, testX, testY, rfc, note)
