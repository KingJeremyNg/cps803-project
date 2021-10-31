import sys
import numpy as np
from read_dataset import readDataset
from linear_model import LinearModel
import logistic_model


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

    print("\nLinear Regression:")
    # Red dataset linear regression
    redLinearModel = LinearModel()
    redLinearModel.fit(redTrainX, redTrainY)
    redPredictions = redLinearModel.predict_array(redTestX)
    compareResults(redPredictions, redTestY, note="red")

    # White dataset linear regression
    whiteLinearModel = LinearModel()
    whiteLinearModel.fit(whiteTrainX, whiteTrainY)
    whitePredictions = whiteLinearModel.predict_array(whiteTestX)
    compareResults(whitePredictions, whiteTestY, note="white")

    print("\nLogistic Regression:")
    # Red dataset logistic regression
    trainY = np.array([1 if x > 5 else 0 for x in redTrainY])
    testY = np.array([1 if x > 5 else 0 for x in redTestY])
    train_x_inter = logistic_model.add_intercept(redTrainX)
    test_x_inter = logistic_model.add_intercept(redTestX)
    classifier = logistic_model.LogisticModel(max_iter=1000, verbose=False)
    classifier.fit(train_x_inter, trainY)
    pred_y_prob = classifier.predict(test_x_inter)
    compareResults([round(x) for x in pred_y_prob], testY, note="red")

    # White dataset logistic regression
    trainY = np.array([1 if x > 5 else 0 for x in whiteTrainY])
    testY = np.array([1 if x > 5 else 0 for x in whiteTestY])
    train_x_inter = logistic_model.add_intercept(whiteTrainX)
    test_x_inter = logistic_model.add_intercept(whiteTestX)
    classifier = logistic_model.LogisticModel(max_iter=1000, verbose=False)
    classifier.fit(train_x_inter, trainY)
    pred_y_prob = classifier.predict(test_x_inter)
    compareResults([round(x) for x in pred_y_prob], testY, note="white")

if __name__ == '__main__':
    main(redDataset='../data/winequality-red.csv',
         whiteDataset='../data/winequality-white.csv')
