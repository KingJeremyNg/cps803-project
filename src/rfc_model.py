import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


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

    #bias_variance(trainX, trainY, testX, testY, rfc, note)
    print(f'CV score: {cross_val_score(rfc, testX, testY, cv=3).mean()}')
