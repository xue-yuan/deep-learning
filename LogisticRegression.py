import numpy as np
from sklearn.preprocessing import StandardScaler


class LogisticRegression:

    def __init__(self, lr=0.001, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        num_samples, num_features = X_scaled.shape

        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.iters):
            z = np.dot(X_scaled, self.weights) + self.bias
            pred = self.sigmoid(z)
            dw = (1 / num_samples) * np.dot(X_scaled.T, (pred - y))
            db = (1 / num_samples) * np.sum(pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = -np.mean(y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))
            self.losses.append(loss)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(z)

        return [1 if p > 0.5 else 0 for p in pred]
