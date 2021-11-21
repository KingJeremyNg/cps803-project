import numpy as np
from compare_results import compareResults
from sklearn import svm


def svmModel(trainX, trainY, testX, testY, note="Unknown"):
    model = svm.SVC()
    trained = model.fit(trainX, trainY)
    prediction = trained.predict(testX)
    compareResults(prediction, testY, note=note)