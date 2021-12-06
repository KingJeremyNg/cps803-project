from compare_results import compareResults
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def mlpGridSearch(trainX, trainY, testX, testY, parameter_space, note="Unknown"):

    mlp = MLPClassifier()
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

    clf.fit(trainX, trainY)
    print('Best parameters found:\n', clf.best_params_)

    pred = clf.predict(testX)
    pred_train = clf.predict(trainX)
    compareResults(pred_train, trainY, note=note + " train")
    compareResults(pred, testY, note=note + " valid")
