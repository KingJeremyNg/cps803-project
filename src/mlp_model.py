from compare_results import compareResults
from sklearn.neural_network import MLPClassifier


def mlpModel(trainX, trainY, testX, testY, note="Unknown"):
    clf = MLPClassifier()
    clf.max_iter = 1000
    clf.fit(trainX, trainY)
    pred = clf.predict(testX)
    pred_train = clf.predict(trainX)

    pred_prob = clf.predict_proba(testX)
    pred_prob_train = clf.predict_proba(trainX)
    labels = clf.classes_

    compareResults(pred_train, trainY, pred_prob_train, labels, note=note + " train")
    compareResults(pred, testY, pred_prob, labels, note=note + " valid")
