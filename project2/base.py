import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class OptimizationResult:
    x: np.ndarray
    fun: float
    niter: int
    success: bool
    message: str
    history: np.ndarray
    grad_norm: float

class ObjectiveFunction(ABC):
    
    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self._check_implementations()
    
    def _check_implementations(self):
        if not hasattr(self, 'value') or not callable(getattr(self, 'value')):
            raise NotImplementedError("value method must be implemented")
        if not hasattr(self, 'gradient') or not callable(getattr(self, 'gradient')):
            raise NotImplementedError("gradient method must be implemented")
    
    @abstractmethod
    def value(self, x: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def hessian(self, x: np.ndarray) -> np.ndarray:
        if self.dimension is None:
            self.dimension = len(x)
        elif len(x) != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {len(x)}")
        
        n = len(x)
        hessian = np.zeros((n, n))
        epsilon = 1e-6
        
        # 用中心差分法计算Hessian矩阵
        for i in range(n):
            for j in range(i, n):
                e_i = np.zeros(n)
                e_i[i] = epsilon
                e_j = np.zeros(n)
                e_j[j] = epsilon
                
                if i == j:
                    # f(x+h*e_i)-2f(x)+f(x-h*e_i))/h²
                    f_plus = self.value(x + e_i)
                    f_minus = self.value(x - e_i)
                    hessian[i, j] = (f_plus - 2 * self.value(x) + f_minus) / (epsilon ** 2)
                else:
                    # [f(x+h*e_i+h*e_j)-f(x+h*e_i-h*e_j)-f(x-h*e_i+h*e_j)+f(x-h*e_i-h*e_j)]/(4h²)
                    f_pp = self.value(x + e_i + e_j)
                    f_pm = self.value(x + e_i - e_j)
                    f_mp = self.value(x - e_i + e_j)
                    f_mm = self.value(x - e_i - e_j)
                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)
                    hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def value_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        return self.value(x), self.gradient(x)
    
    def value_gradient_hessian(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.value(x), self.gradient(x), self.hessian(x)

class Optimizer(ABC):
    
    def __init__(self, tol: float = 1e-6, max_iter: int = 1000, 
                 verbose: bool = False, history: bool = True,
                 callback: Optional[callable] = None):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.store_history = history
        self.callback = callback
        self.history: List[np.ndarray] = []
    
    @abstractmethod
    def minimize(self, func: ObjectiveFunction, x0: np.ndarray) -> OptimizationResult:
        pass
    
    def _store_history(self, x: np.ndarray): # 迭代历史存储
        if self.store_history:
            self.history.append(x.copy())
    
    def _check_convergence(self, gradient: np.ndarray, iteration: int) -> bool:
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < self.tol:
            if self.verbose:
                print(f"Optimization converged at iteration {iteration}, ||g|| = {grad_norm:.2e}")
            return True
        return False
    
    def _print_iteration(self, iteration: int, func_value: float, grad_norm: float):
        if self.verbose and iteration % 10 == 0:
            print(f"Iter {iteration:4d}: f(x) = {func_value:.6e}, ||g|| = {grad_norm:.6e}")
    
    def _execute_callback(self, iteration: int, x: np.ndarray, 
                         func_value: float, gradient: np.ndarray):
        if self.callback:
            self.callback({
                'iteration': iteration,
                'x': x.copy(),
                'fun': func_value,
                'grad': gradient.copy(),
                'grad_norm': np.linalg.norm(gradient)
            })
    
    def _validate_input(self, func: ObjectiveFunction, x0: np.ndarray):
        if not isinstance(func, ObjectiveFunction):
            raise TypeError("func must be an instance of ObjectiveFunction")
        
        x0 = np.asarray(x0)
        if x0.ndim != 1:
            raise ValueError("x0 must be a 1-dimensional array")
        
        try:
            test_value = func.value(x0)
            test_gradient = func.gradient(x0)
            if not np.isscalar(test_value):
                raise ValueError("func.value() must return a scalar")
            if test_gradient.shape != x0.shape:
                raise ValueError("func.gradient() must return array with same shape as input")
        except Exception as e:
            raise ValueError(f"Function evaluation failed: {e}") from e

# 为二阶优化算法提供的额外工具函数
def is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def modify_to_positive_definite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    n = matrix.shape[0]
    min_eigval = np.min(np.linalg.eigvalsh(matrix))
    
    if min_eigval > 0:
        return matrix
    else:
        return matrix + (abs(min_eigval) + epsilon) * np.eye(n)

def check_descent_direction(gradient: np.ndarray, direction: np.ndarray) -> bool:
    return np.dot(gradient, direction) < 0