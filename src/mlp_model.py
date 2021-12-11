from compare_results import compareResults, bias_variance
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def mlpModel(trainX, trainY, testX, testY, note="Unknown"):
    clf = MLPClassifier()
    clf.max_iter = 5000
    clf.fit(trainX, trainY)
    pred = clf.predict(testX)
    pred_train = clf.predict(trainX)

    pred_prob = clf.predict_proba(testX)
    pred_prob_train = clf.predict_proba(trainX)
    labels = clf.classes_

    compareResults(pred_train, trainY, pred_prob_train, labels, note=note + " train")
    compareResults(pred, testY, pred_prob, labels, note=note + " valid")

    #bias_variance(trainX, trainY, testX, testY, clf, note)
    print(f'CV score: {cross_val_score(clf, testX, testY, cv=3).mean()}')
