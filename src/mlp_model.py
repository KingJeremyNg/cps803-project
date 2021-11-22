from compare_results import compareResults
from sklearn.neural_network import MLPClassifier


def mlpModel(trainX, trainY, testX, testY, note="Unknown"):
    clf = MLPClassifier()
    clf.max_iter = 1000
    clf.fit(trainX, trainY)
    pred = clf.predict(testX)
    compareResults(pred, testY, note=note)
