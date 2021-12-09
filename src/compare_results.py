import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.evaluate import bias_variance_decomp


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
    bias = np.mean(trueLabels-predictions)
    print(f"{note} success rate: {success_rate}")
    #print(f"{note} Variance: {np.var(predictions)}")
    #print(f"{note} Average Bias: {bias}")
    # print(f"{note} average loss: {average_loss}")
    # print("Confusion matrix:")
    # print(f"Labels: {set(trueLabels)}")
    # print(confusion_matrix(trueLabels, predictions))
    return success_rate

def bias_variance(X_train, y_train, X_test, y_test, model, note):
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, X_train, y_train, X_test, y_test, 
        loss='0-1_loss',
        random_seed=123,
        num_rounds=100)
    print(f'{note} Average Expected Loss: {round(avg_expected_loss, 4)}')
    print(f'{note} Average Bias: {round(avg_bias, 4)}')
    print(f'{note} Average Variance: {round(avg_var, 4)}\n')
