import numpy as np


class LinearRegressionGD:
    def __init__(
        self,
        learning_rate=0.01,
        max_iters=1000,
        tol=1e-6,
        init_method="zeros",
        random_scale=0.01,
    ):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.random_scale = random_scale

        self.w = None
        self.b = None
        self.loss_history = []

    def _initialize_params(self, n_features):
        if self.init_method == "random":
            self.w = np.random.randn(n_features) * self.random_scale
            self.b = 0.0
        else:
            self.w = np.zeros(n_features)
            self.b = 0.0

    def predict(self, X):
        return X @ self.w + self.b

    def compute_cost(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def compute_gradients(self, X, y, y_hat):
        n = len(y)
        residuals = y_hat - y
        dw = (2 / n) * X.T @ residuals
        db = (2 / n) * np.sum(residuals)
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_params(n_features)

        prev_loss = float("inf")

        for _ in range(self.max_iters):
            y_hat = self.predict(X)
            loss = self.compute_cost(y_hat, y)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break

            dw, db = self.compute_gradients(X, y, y_hat)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            prev_loss = loss

        return self

