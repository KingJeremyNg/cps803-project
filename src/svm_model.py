import numpy as np
from compare_results import compareResults, bias_variance
from sklearn import svm
from sklearn.model_selection import cross_val_score


def svmModel(trainX, trainY, testX, testY, note="Unknown"):
    model = svm.SVC(probability=True)
    trained = model.fit(trainX, trainY)

    prediction = trained.predict(testX)
    prediction_train = trained.predict(trainX)

    pred_prob = model.predict_proba(testX)
    pred_prob_train = model.predict_proba(trainX)
    labels = model.classes_

    compareResults(prediction_train, trainY, pred_prob_train, labels, note=note + " train")

    #bias_variance(trainX, trainY, testX, testY, trained, note)
    print(f'CV score: {cross_val_score(model, testX, testY, cv=3).mean()}')
