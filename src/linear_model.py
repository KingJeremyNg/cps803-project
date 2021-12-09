import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.linear_model import LinearRegression


def linearModel(trainX, trainY, testX, testY, note="Unknown"):
    reg = LinearRegression().fit(trainX, trainY)
    prediction = [round(x) for x in reg.predict(testX)]
    prediction_train = [round(x) for x in reg.predict(trainX)]
    compareResults(np.array(prediction_train), trainY, note=(note + ' train'))
    compareResults(np.array(prediction), testY, note=(note + " valid"))
    #bias_variance(trainX, trainY, testX, testY, reg, note)
