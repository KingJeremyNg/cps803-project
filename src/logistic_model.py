import numpy as np
from compare_results import compareResults
from sklearn.linear_model import LogisticRegression


def logisticModel(trainX, trainY, testX, testY, note="Unknown"):
    logTrainY = [1 if x > 5 else 0 for x in trainY]
    logTestY = [1 if x > 5 else 0 for x in testY]
    log = LogisticRegression()
    log.max_iter = 10000
    log.fit(trainX, logTrainY)
    prediction = [round(x) for x in log.predict(testX)]
    compareResults(np.array(prediction), np.array(logTestY), note=note)
