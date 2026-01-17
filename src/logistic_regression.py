import numpy as np
from .losses import binary_log_loss


class LogisticRegressionGD:
    def __init__(
        self,
        learning_rate=0.1,
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

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """
        Compute accuracy on any dataset.
        Evaluation logic is intentionally decoupled from training.
        """
        return np.mean(self.predict(X) == y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_params(n_features)

        prev_loss = float("inf")

        for _ in range(self.max_iters):
            y_hat = self.predict_proba(X)
            loss = binary_log_loss(y, y_hat)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break

            dw = (1 / n_samples) * X.T @ (y_hat - y)
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            prev_loss = loss

        return self
