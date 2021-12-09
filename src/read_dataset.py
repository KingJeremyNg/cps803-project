import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def readDataset(path):
    X = pd.read_csv(path, usecols=[i for i in range(11)]).to_numpy()
    y = pd.read_csv(path, usecols=["quality"]).to_numpy()
    trainX, testX, trainY, testY = train_test_split(
        X, y, test_size=0.1, random_state=0)
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.fit_transform(testX)
    # train_test_split is giving dim(n, 1) when it should be dim(n,)

    # Finding the index of outliers in categories with ourliers using the winsorization method
    va_out = winsorization_outliers(trainX[:,1], "volitile acidity", True, True)
    rs_out = winsorization_outliers(trainX[:,5], "residual sugar", False, True)
    ch_out = winsorization_outliers(trainX[:,6], "chlorides", False, True)
    sl_out = winsorization_outliers(trainX[:,9], "sulphates", True, True)
    
    union = np.union1d(rs_out, ch_out)
    union = np.union1d(union, sl_out)
    union = np.union1d(union, va_out)

    trainX = np.delete(trainX, union, axis=0)
    trainY = np.delete(trainY, union, axis=0)

    #trainX, trainY = remove_rows(trainX, trainY, union)

    return trainX, trainY[:, 0], testX, testY[:, 0]

# top_percent = boolean to determine if highest percentile should be removed
# bottom_percent = boolean to determine if lowest percentile should be removed
def  winsorization_outliers(df, label, top_percent, bottom_percent): 
    q1 = np.percentile(df , 1)
    q3 = np.percentile(df , 99)
    out = []
    out_map = {}
    for i in range(len(df)):
        if (df[i] > q3 and top_percent) or (df[i] < q1 and bottom_percent):
            out.append(i)
            out_map[i]=df[i]
    #print("Outliers for label" + label + ": ",out2)
    return out
    