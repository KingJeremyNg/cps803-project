import numpy as np
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
    print(f"{note} success rate: {success_rate}")
    # print(f"{note} average loss: {average_loss}")
    # print("Confusion matrix:")
    # print(f"Labels: {set(trueLabels)}")
    # print(confusion_matrix(trueLabels, predictions))
    return success_rate
