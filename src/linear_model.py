import numpy as np
from compare_results import compareResults
from sklearn.linear_model import LinearRegression


def linearModel(trainX, trainY, testX, testY, note="Unknown"):
    reg = LinearRegression().fit(trainX, trainY)
    prediction = [round(x) for x in reg.predict(testX)]
    compareResults(np.array(prediction), testY, note=note)
