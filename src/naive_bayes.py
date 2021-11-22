import numpy as np
from compare_results import compareResults
from sklearn.naive_bayes import GaussianNB


def naiveBayes(trainX, trainY, testX, testY, note="Unknown"):
    bayes = GaussianNB().fit(trainX, trainY)
    prediction = bayes.predict(testX)
    compareResults(prediction, testY, note=note)
