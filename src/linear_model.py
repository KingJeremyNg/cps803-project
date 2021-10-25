import numpy as np

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(X.T @ X,  X.T) @ y
        # self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        # *** END CODE HERE ***

    def fit_GD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        dimensions = X.shape[1]
        m = X.shape[0]
        print(f"dim: {dimensions}")
        print(f"iterations: {ITERATIONS}")
        print(f"inputs: {m}")
        self.theta = np.zeros([dimensions])
        

        # iterations
        for iteration in range(ITERATIONS):
            for j in range(dimensions):
                diff = 0
                for i in range(m):
                    temp_sum = 0
                    for k in range(dimensions):
                        temp_sum += self.theta[k]*X[i][k]
                    diff += (temp_sum - y[i])*X[i][j]
                self.theta[j] = self.theta[j] - LEARNING_RATE * diff

        # *** END CODE HERE ***

    def fit_SGD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        dimensions = X.shape[1]
        m = X.shape[0]
        print(f"dim: {dimensions}")
        print(f"iterations: {ITERATIONS}")
        print(f"inputs: {m}")
        self.theta = np.zeros([dimensions])

        # iterations
        for iteration in range(ITERATIONS):
            for i in range(m):
                for j in range(dimensions):
                    diff = 0
                    for k in range(dimensions):
                        diff += self.theta[k]*X[i][k]
                    diff = (diff - y[i])*X[i][j]
                    self.theta[j] = self.theta[j] - LEARNING_RATE * diff
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        numOfInputs = k+1
        feature_map = np.ones(X.shape[0])
        # Adding x^2, x^3... dimensions to training samples
        for i in range(1, numOfInputs):
            feature_map = np.c_[feature_map, X[:, 1]**i]

        return feature_map


        # *** END CODE HERE ***

    def create_cosine(self, k, X):
        """
        Generates a cosine with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        numOfInputs = k+1
        feature_map = np.ones(X.shape[0])

        # Adding x^2, x^3... dimensions to training samples
        for i in range(1, numOfInputs):
            feature_map = np.c_[feature_map, X[:, 1]**i]

        feature_map = np.c_[feature_map, np.cos(X[:, 1])]

        return feature_map
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return X.dot(self.theta)
        # *** END CODE HERE ***

    def predict_array(self, arrayX):
        results = []
        for n in range(len(arrayX)):
            prediction = int(round(self.predict(arrayX[n]),0))
            results.append(prediction)
        return results
       