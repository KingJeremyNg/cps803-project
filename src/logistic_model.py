import numpy as np
from compare_results import compareResults, bias_variance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def logisticModel(trainX, trainY, testX, testY, note="Unknown"):
    log = LogisticRegression(multi_class='multinomial')
    log.max_iter = 10000
    log.fit(trainX, trainY)
    prediction = [round(x) for x in log.predict(testX)]
    prediction_train = [round(x) for x in log.predict(trainX)]
    compareResults(np.array(prediction_train),
                   np.array(trainY), note=note + " train")
    compareResults(np.array(prediction), np.array(testY), note=note + " valid")
    #bias_variance(trainX, trainY, testX, testY, log, note)
    print(f'CV score: {cross_val_score(log, testX, testY, cv=3).mean()}')
