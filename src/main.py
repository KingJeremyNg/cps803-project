import sys
import numpy as np
from read_dataset import readDataset
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def compareResults(predictions, trueLabels, note="-"):
    if (len(predictions) != len(trueLabels)):
        raise ValueError(
            "The lengths of predictions and trueLabels are not the same."
        )
    if (not (type(predictions) is np.ndarray and type(trueLabels) is np.ndarray)):
        print(f"predictions type: {type(predictions)}")
        print(f"trueLabels type: {type(trueLabels)}")
        raise TypeError(
            "The type of predictions and trueLabels must be numpy array."
        )
    success_rate = np.mean(predictions == trueLabels)
    average_loss = np.mean(np.absolute(trueLabels - predictions))
    print(f"{note} success rate: {success_rate}.")
    print(f"{note} average loss: {average_loss}.")
    print("Confusion matrix:")
    print(f"Labels: {set(trueLabels)}")
    print(confusion_matrix(trueLabels, predictions))
    return success_rate


def linearModel(trainX, trainY, testX, testY, note="Unknown"):
    reg = LinearRegression().fit(trainX, trainY)
    prediction = [round(x) for x in reg.predict(testX)]
    compareResults(np.array(prediction), testY, note=note)


def logisticModel(trainX, trainY, testX, testY, note="Unknown"):
    logTrainY = [1 if x > 5 else 0 for x in trainY]
    logTestY = [1 if x > 5 else 0 for x in testY]
    log = LogisticRegression()
    log.max_iter = 1000
    log.fit(trainX, logTrainY)
    prediction = [round(x) for x in log.predict(testX)]
    compareResults(np.array(prediction), np.array(logTestY), note=note)


def naiveBayes(trainX, trainY, testX, testY, note="Unknown"):
    bayes = GaussianNB().fit(trainX, trainY)
    prediction = bayes.predict(testX)
    compareResults(prediction, testY, note=note)


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier().fit(trainX, trainY)
    prediction = knn.predict(testX)
    compareResults(prediction, testY, note=note)


def svmModel(trainX, trainY, testX, testY, note="Unknown"):
    model = svm.SVC()
    trained = model.fit(trainX, trainY)
    prediction = trained.predict(testX)
    compareResults(prediction, testY, note=note)


def rfcModel(trainX, trainY, testX, testY, note="Unknown"):
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(trainX, trainY)
    pred = rfc.predict(testX)
    compareResults(pred, testY, note=note)


def linearModel(trainX, trainY, testX, testY, note="Unknown"):
    reg = LinearRegression().fit(trainX, trainY)
    prediction = [round(x) for x in reg.predict(testX)]
    compareResults(prediction, testY, note=note)


def logisticModel(trainX, trainY, testX, testY, note="Unknown"):
    logTrainY = [1 if x > 5 else 0 for x in trainY]
    logTestY = [1 if x > 5 else 0 for x in testY]
    log = LogisticRegression()
    log.max_iter = 1000
    log.fit(trainX, logTrainY)
    prediction = [round(x) for x in log.predict(testX)]
    compareResults(prediction, logTestY, note=note)


def naiveBayes(trainX, trainY, testX, testY, note="Unknown"):
    bayes = GaussianNB().fit(trainX, trainY)
    prediction = bayes.predict(testX)
    compareResults(prediction, testY, note=note)


def knn(trainX, trainY, testX, testY, note="Unknown"):
    knn = KNeighborsClassifier().fit(trainX, trainY)
    prediction = knn.predict(testX)
    compareResults(prediction, testY, note=note)


def main(redDataset, whiteDataset):
    # Load dataset
    redFeatures, redResult = readDataset(redDataset)
    whiteFeatures, whiteResult = readDataset(whiteDataset)
    trainSetPercentage = 0.8

    # Red dataset
    redLength = len(redFeatures)
    redIndex = round(trainSetPercentage*redLength)
    redTrainX = redFeatures[0:redIndex]
    redTrainY = redResult[0:redIndex]
    redTestX = redFeatures[redIndex:redLength]
    redTestY = redResult[redIndex:redLength]

    # White dataset
    whiteLength = len(whiteFeatures)
    whiteIndex = round(trainSetPercentage*whiteLength)
    whiteTrainX = whiteFeatures[0:whiteIndex]
    whiteTrainY = whiteResult[0:whiteIndex]
    whiteTestX = whiteFeatures[whiteIndex:whiteLength]
    whiteTestY = whiteResult[whiteIndex:whiteLength]

    # Linear Regression for Red and White Datasets
    print("\n==========================================\nLinear Regression:")
    linearModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    linearModel(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # Logistic Regression for Red and White Datasets
    print("\n==========================================\nLogistic Regression:")
    logisticModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    logisticModel(whiteTrainX, whiteTrainY,
                  whiteTestX, whiteTestY, note="White")

    # Naive Bayes for Red and White Datasets
    print("\n==========================================\nNaive Bayes:")
    naiveBayes(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    naiveBayes(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # KNN for Red and White Datasets
    print("\n==========================================\nK-Nearest Neighbours:")
    knn(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    knn(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # SVM for Red and White Datasets
    print("\n==========================================\nSupport Vector Model:")
    svmModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    svmModel(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # RFC (Random Forest Classifier) on Red and White Datasets
    print("\n==========================================\nRandom Forest Classifier:")
    rfcModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    rfcModel(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")


if __name__ == '__main__':
    main(redDataset='../data/winequality-red.csv',
         whiteDataset='../data/winequality-white.csv')
