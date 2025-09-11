import numpy as np

class Problem:
    def dim(self):
        raise NotImplementedError

    def loss(self, w):
        raise NotImplementedError

    def gradient(self, w):
        raise NotImplementedError

    def stochastic_gradient(self, w, idx):
        raise NotImplementedError

    def minibatch_gradient(self, w, idx_list):
        raise NotImplementedError


# ============ Quadratic Problem (Least Squares) ============
class QuadraticProblem(Problem):
    def __init__(self, A, b):
        # Use float32 for speed/memory unless caller requires higher precision
        self.A = np.asarray(A, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.n, self.d = self.A.shape

    def dim(self):
        return self.d

    def loss(self, w):
        w = w.astype(np.float32, copy=False)
        r = self.A @ w - self.b
        # 0.5 * ||Aw - b||^2
        return 0.5 * float(r @ r)

    def gradient(self, w):
        w = w.astype(np.float32, copy=False)
        return self.A.T @ (self.A @ w - self.b)

    def stochastic_gradient(self, w, idx):
        w = w.astype(np.float32, copy=False)
        xi = self.A[idx]
        yi = self.b[idx]
        return xi * (xi @ w - yi)

    def minibatch_gradient(self, w, idx_list):
        w = w.astype(np.float32, copy=False)
        Xb = self.A[idx_list]
        yb = self.b[idx_list]
        return (Xb.T @ (Xb @ w - yb)) / np.float32(len(idx_list))


# ============ Logistic Regression Problem ============
class LogisticProblem(Problem):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        # Ensure labels are in {0,1} float32
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
        self.y = y
        self.n, self.d = self.X.shape

    def dim(self):
        return self.d

    @staticmethod
    def sigmoid(z):
        # Numerically stable sigmoid
        # For z >= 0: 1/(1+exp(-z)) ; for z < 0: exp(z)/(1+exp(z))
        out = np.empty_like(z, dtype=np.float32)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos], dtype=np.float32))
        ez = np.exp(z[neg], dtype=np.float32)
        out[neg] = ez / (1.0 + ez)
        return out

    def loss(self, w):
        w = w.astype(np.float32, copy=False)
        z = self.X @ w
        # Logistic loss: mean( log(1 + exp(z)) - y*z )
        # Use logaddexp for stability
        return float(np.mean(np.logaddexp(0.0, z) - self.y * z))

    def gradient(self, w):
        w = w.astype(np.float32, copy=False)
        z = self.X @ w
        yhat = self.sigmoid(z)
        return (self.X.T @ (yhat - self.y)) / np.float32(self.n)

    def stochastic_gradient(self, w, idx):
        w = w.astype(np.float32, copy=False)
        xi = self.X[idx]
        yi = self.y[idx]
        yhat = 1.0 / (1.0 + np.exp(-xi @ w, dtype=np.float32))
        return xi * (yhat - yi)

    def minibatch_gradient(self, w, idx_list):
        w = w.astype(np.float32, copy=False)
        Xb = self.X[idx_list]
        yb = self.y[idx_list]
        z = Xb @ w
        yhat = self.sigmoid(z)
        return (Xb.T @ (yhat - yb)) / np.float32(len(idx_list))
