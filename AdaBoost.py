import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.p = 1
        self.alpha = None

    def fit(self, X, y, weights):
        m, n = X.shape
        min_error = float("inf")

        for feature in range(n):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                p = 1
                predictions = np.ones(m)
                predictions[X[:, feature] < threshold] = -1

                error = np.sum(weights[y != predictions])

                if error > 0.5:
                    error = 1 - error
                    p = -1

                if error < min_error:
                    self.p = p
                    self.threshold = threshold
                    self.feature = feature
                    min_error = error

        return min_error

    def predict(self, X):
        n = X.shape[0]
        predictions = np.ones(n)
        if self.p == 1:
            predictions[X[:, self.feature] < self.threshold] = -1
        else:
            predictions[X[:, self.feature] >= self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, T=100):
        self.T = T
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        m = X.shape[0]
        weights = np.ones(m) / m

        for t in range(self.T):
            stump = DecisionStump()
            error = stump.fit(X, y, weights)

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            self.stumps.append(stump)

            predictions = stump.predict(X)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            print(
                f"Iteration {t+1}: Feature {stump.feature}, Threshold {stump.threshold}, C1 {1 if stump.p == 1 else -1}"
            )

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            final_predictions += alpha * stump.predict(X)
        return np.sign(final_predictions)
