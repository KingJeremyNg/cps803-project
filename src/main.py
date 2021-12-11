import numpy as np
from read_dataset import readDataset
from linear_model import linearModel
from logistic_model import logisticModel
from naive_bayes import naiveBayes
from knn_model import knn
from svm_model import svmModel
from rfc_model import rfcModel
from mlp_model import mlpModel
from mlp_grid_search import mlpGridSearch


def main(redPath, whitePath):
    # Red dataset
    redTrainX, redTrainY, redTestX, redTestY = readDataset(redPath)
    # print(redTrainX.shape, redTrainY.shape)
    # print(redTestX.shape, redTestY.shape)

    # White dataset
    whiteTrainX, whiteTrainY, whiteTestX, whiteTestY = readDataset(whitePath)
    # print(whiteTrainX.shape, whiteTrainY.shape)
    # print(whiteTestX.shape, whiteTestY.shape)

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

    # Multi Layer Perception (Neural Network) on Red and White Datasets
    print("\n==========================================\nMulti Layer Perception Neural Network:")
    mlpModel(redTrainX, redTrainY, redTestX, redTestY, note="Red")
    mlpModel(whiteTrainX, whiteTrainY, whiteTestX, whiteTestY, note="White")

    # MLP (Neural Network) GridSearch on Red and White Datasets
    print("\n==========================================\nMLP Grid Search:")
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 100, 100), (16, 32, 16), (100, 100, 100, 100, 100), (100,)],
        'activation': ['tanh', 'relu', "logistic", "identity"],
        "max_iter": [1200],
        'alpha': [0.0001, 0.01, 0.001, 0.05],
        'learning_rate': ['adaptive', 'constant'],
        'solver': ['sgd', 'adam'],
    }
    mlpGridSearch(redTrainX, redTrainY, redTestX,
                  redTestY, parameter_space, note="Red")
    mlpGridSearch(whiteTrainX, whiteTrainY, whiteTestX,
                  whiteTestY, parameter_space, note="White")


if __name__ == '__main__':
    main(redPath='../data/winequality-red.csv',
         whitePath='../data/winequality-white.csv')
