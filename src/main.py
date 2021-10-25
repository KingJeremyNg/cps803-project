import sys
import readDataset as reader
import numpy as np
import linear_model

LinearModel = linear_model.LinearModel


def main(redDataset, whiteDataset):
    redFeatures, redResult = reader.readDataset(redDataset)
    whiteFeatures, whiteResult = reader.readDataset(whiteDataset)

    redLength = len(redFeatures)
    redIndex = round(0.8*redLength)
    redTrainX = redFeatures[0:redIndex]
    redTrainY = redResult[0:redIndex]
    redTestX = redFeatures[redIndex:redLength]
    redTestY = redResult[redIndex:redLength]

    print(redTrainX)
    linearModel = LinearModel()
    linearModel.fit(redTrainX, redTrainY)
    for n in range(len(redTestX)):
        prediction = int(round(linearModel.predict(redTestX[n]),0))
        trueLabel = redTestY[n]
        same = prediction == trueLabel
        print(f"prediction: {prediction}, true: {trueLabel}, same: {same}")

    whiteLength = len(whiteFeatures)
    whiteIndex = round(0.8*whiteLength)
    whiteTrainX = whiteFeatures[0:whiteIndex]
    whiteTrainY = whiteResult[0:whiteIndex]
    whiteTestX = whiteFeatures[whiteIndex:whiteLength]
    whiteTestY = whiteResult[whiteIndex:whiteLength]


if __name__ == '__main__':
    main(redDataset='../data/winequality-red.csv',
         whiteDataset='../data/winequality-white.csv')
