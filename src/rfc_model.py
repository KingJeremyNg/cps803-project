import numpy as np
from compare_results import compareResults
from sklearn.ensemble import RandomForestClassifier


def rfcModel(trainX, trainY, testX, testY, note="Unknown"):
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(trainX, trainY)
    pred = rfc.predict(testX)
    compareResults(pred, testY, note=note)
