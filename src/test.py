import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

path = '../data/winequality-red.csv'

X = pd.read_csv(path, usecols=[i for i in range(11)]).to_numpy()
y = pd.read_csv(path, usecols=["quality"]).to_numpy()
trainX, testX, trainY, testY = train_test_split(
    X, y, test_size=0.2, random_state=0)

reg = LinearRegression()
print(cross_val_score(reg, X, y, cv=10).mean())
print(cross_val_score(reg, trainX, trainY, cv=10).mean())
print(cross_val_score(reg, testX, testY, cv=10).mean())
