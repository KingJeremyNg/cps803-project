import sys
import numpy as np
from read_dataset import readDataset
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def compareResults(predictions, trueLabels, verbose=False, note="Unknown"):
    if (len(predictions) != len(trueLabels)):
        raise ValueError(
            "The lengths of predictions and trueLabels are not the same.")
    numSuccess = 0
    for n in range(len(predictions)):
        prediction = predictions[n]
        trueLabel = trueLabels[n]
        success = prediction == trueLabel
        if success:
            numSuccess += 1
        if (verbose):
            print(
                f"prediction: {prediction}, true label: {trueLabel}, success: {success}")
    print(f"{note} success rate: {round((100*numSuccess)/len(predictions), 2)}%.")


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

    # Red dataset
    redLength = len(redFeatures)
    redIndex = round(0.8*redLength)
    redTrainX = redFeatures[0:redIndex]
    redTrainY = redResult[0:redIndex]
    redTestX = redFeatures[redIndex:redLength]
    redTestY = redResult[redIndex:redLength]

    # White dataset
    whiteLength = len(whiteFeatures)
    whiteIndex = round(0.8*whiteLength)
    whiteTrainX = whiteFeatures[0:whiteIndex]
    whiteTrainY = whiteResult[0:whiteIndex]
    whiteTestX = whiteFeatures[whiteIndex:whiteLength]
    whiteTestY = whiteResult[whiteIndex:whiteLength]

    # Linear Regression for Red and White Datasets
    print("\nLinear Regression:")
    linearModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    linearModel(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # Logistic Regression for Red and White Datasets
    print("\nLogistic Regression:")
    logisticModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    logisticModel(whiteTrainX, whiteTrainY,
                  whiteTestX, whiteTestY, note="White")

    # Naive Bayes for Red and White Datasets
    print("\nNaive Bayes:")
    naiveBayes(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    naiveBayes(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # KNN for Red and White Datasets
    print("\nK-Nearest Neighbours:")
    knn(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    knn(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")


if __name__ == '__main__':
    main(redDataset='../data/winequality-red.csv',
         whiteDataset='../data/winequality-white.csv')
