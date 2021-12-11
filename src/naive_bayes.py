import numpy as np
from compare_results import compareResults
from sklearn.naive_bayes import GaussianNB


def naiveBayes(trainX, trainY, testX, testY, note="Unknown"):
    bayes = GaussianNB().fit(trainX, trainY)
    prediction = bayes.predict(testX)
    prediction_train = bayes.predict(trainX)

    pred_prob = bayes.predict_proba(testX)
    pred_prob_train = bayes.predict_proba(trainX)
    labels = bayes.classes_
    
    compareResults(prediction_train, trainY, pred_prob_train,labels, note=note + " train")
    compareResults(prediction, testY, pred_prob, labels,note=note + " valid")
