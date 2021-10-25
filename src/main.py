import sys
import readDataset as reader
import numpy as np
import linear_model

LinearModel = linear_model.LinearModel

def compareResults(predictions, trueLabels, verbose=False, note="Unknown"):
    if (len(predictions) != len(trueLabels)):
        raise ValueError("The lengths of predictions and trueLabels are not the same.")
    numSuccess = 0
    for n in range(len(predictions)):
        prediction = predictions[n]
        trueLabel = trueLabels[n]
        success = prediction == trueLabel
        if success:
            numSuccess += 1
        if (verbose):
            print(f"prediction: {prediction}, true label: {trueLabel}, success: {success}")
    print(f"{note} success rate: {100*numSuccess/len(predictions)}%.")

def main(redDataset, whiteDataset):
    redFeatures, redResult = reader.readDataset(redDataset)
    whiteFeatures, whiteResult = reader.readDataset(whiteDataset)

    redLength = len(redFeatures)
    redIndex = round(0.8*redLength)
    redTrainX = redFeatures[0:redIndex]
    redTrainY = redResult[0:redIndex]
    redTestX = redFeatures[redIndex:redLength]
    redTestY = redResult[redIndex:redLength]

    redLinearModel = LinearModel()
    redLinearModel.fit(redTrainX, redTrainY)
    redPredictions = redLinearModel.predict_array(redTestX)
    compareResults(redPredictions, redTestY, note="red")
        

    whiteLength = len(whiteFeatures)
    whiteIndex = round(0.8*whiteLength)
    whiteTrainX = whiteFeatures[0:whiteIndex]
    whiteTrainY = whiteResult[0:whiteIndex]
    whiteTestX = whiteFeatures[whiteIndex:whiteLength]
    whiteTestY = whiteResult[whiteIndex:whiteLength]

    whiteLinearModel = LinearModel()
    whiteLinearModel.fit(whiteTrainX, whiteTrainY)
    whitePredictions = whiteLinearModel.predict_array(whiteTestX)
    compareResults(whitePredictions, whiteTestY, note="white")


if __name__ == '__main__':
    main(redDataset='../data/winequality-red.csv',
         whiteDataset='../data/winequality-white.csv')
