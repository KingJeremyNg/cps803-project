from compare_results import compareResults, bias_variance
from sklearn.neural_network import MLPClassifier


def mlpModel(trainX, trainY, testX, testY, note="Unknown"):
    clf = MLPClassifier()
    clf.max_iter = 5000
    clf.fit(trainX, trainY)
    pred = clf.predict(testX)
    pred_train = clf.predict(trainX)
    compareResults(pred_train, trainY, note=note + " train")
    compareResults(pred, testY, note=note + " valid")
    #bias_variance(trainX, trainY, testX, testY, clf, note)
