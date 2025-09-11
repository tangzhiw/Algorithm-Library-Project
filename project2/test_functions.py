import numpy as np
from base import ObjectiveFunction

class QuadraticFunction(ObjectiveFunction):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        super().__init__(dimension=len(b))
        self.A = A
        self.b = b
        if A.shape != (len(b), len(b)):
            raise ValueError("Matrix A must be square and match dimension of b")
    
    def value(self, x: np.ndarray) -> float:
        if len(x) != self.dimension:
            raise ValueError(f"Input dimension mismatch: expected {self.dimension}, got {len(x)}")
        return 0.5 * x.T @ self.A @ x - self.b.T @ x
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        if len(x) != self.dimension:
            raise ValueError(f"Input dimension mismatch: expected {self.dimension}, got {len(x)}")
        return self.A @ x - self.b
    
    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A

class RosenbrockFunction(ObjectiveFunction):
    def __init__(self):
        super().__init__(dimension=2)
    
    def value(self, x: np.ndarray) -> float:
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2-dimensional input")
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2-dimensional input")
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])
    
    def hessian(self, x: np.ndarray) -> np.ndarray:
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2-dimensional input")
        return np.array([
            [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])

class LogisticRegressionFunction(ObjectiveFunction):   
    def __init__(self, X: np.ndarray, y: np.ndarray, lambda_reg: float = 0.1):
        self.X = X
        self.y = y
        self.lambda_reg = lambda_reg
        super().__init__(dimension=X.shape[1])
    
    def value(self, w: np.ndarray) -> float:
        if len(w) != self.dimension:
            raise ValueError(f"Weight dimension mismatch: expected {self.dimension}, got {len(w)}")
        z = self.X @ w
        logistic = 1 / (1 + np.exp(-z))
        loss = -np.mean(self.y * np.log(logistic + 1e-15) + 
                       (1 - self.y) * np.log(1 - logistic + 1e-15))
        regularization = 0.5 * self.lambda_reg * np.sum(w**2)
        
        return loss + regularization
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        if len(w) != self.dimension:
            raise ValueError(f"Weight dimension mismatch: expected {self.dimension}, got {len(w)}")
        z = self.X @ w
        logistic = 1 / (1 + np.exp(-z))
        grad = self.X.T @ (logistic - self.y) / len(self.y)
        grad_reg = self.lambda_reg * w
        
        return grad + grad_reg
    
    def hessian(self, w: np.ndarray) -> np.ndarray:
        if len(w) != self.dimension:
            raise ValueError(f"Weight dimension mismatch: expected {self.dimension}, got {len(w)}")

        z = self.X @ w
        logistic = 1 / (1 + np.exp(-z))
        D = np.diag(logistic * (1 - logistic))
        hessian = self.X.T @ D @ self.X / len(self.y)
        hessian_reg = self.lambda_reg * np.eye(self.dimension)
        
        return hessian + hessian_reg