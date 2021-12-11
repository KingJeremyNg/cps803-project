import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from read_dataset import readDataset

path = '../data/winequality-red.csv'

trainX, trainY, testX, testY = readDataset(path)

reg = LinearRegression().fit(trainX, trainY)
prediction = [round(x) for x in reg.predict(testX)]
prediction_train = [round(x) for x in reg.predict(trainX)]

print(cross_val_score(reg, trainX, trainY, cv=3).mean())
print(cross_val_score(reg, testX, testY, cv=3).mean())
