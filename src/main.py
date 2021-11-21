import sys
import numpy as np
from read_dataset import readDataset
from linear_model import linearModel
from logistic_model import logisticModel
from naive_bayes import naiveBayes
from knn_model import knn
from svm_model import svmModel
from rfc_model import rfcModel


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
