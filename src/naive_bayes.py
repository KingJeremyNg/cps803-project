import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.naive_bayes import GaussianNB


def naiveBayes(trainX, trainY, testX, testY, note="Unknown"):
    bayes = GaussianNB().fit(trainX, trainY)
    prediction = bayes.predict(testX)
    prediction_train = bayes.predict(trainX)
    compareResults(prediction_train, trainY, note=note + " train")
    compareResults(prediction, testY, note=note + " valid")
    #bias_variance(trainX, trainY, testX, testY, bayes, note)
