import numpy as np
from scipy.linalg import solve
from base import Optimizer, OptimizationResult, modify_to_positive_definite
from line_search import backtracking_line_search


class NewtonMethod(Optimizer):
    
    def __init__(self, tol: float = 1e-6, max_iter: int = 100, 
                 verbose: bool = False, history: bool = True,
                 line_search: bool = True, modify_hessian: bool = True):
        super().__init__(tol, max_iter, verbose, history)
        self.line_search = line_search
        self.modify_hessian = modify_hessian
    
    def minimize(self, func, x0):
        self._validate_input(func, x0)
        x = np.asarray(x0, dtype=float).copy()
        self.history = []
        self._store_history(x)
        message = "Optimization completed"
        
        for i in range(self.max_iter):
            try:
                # 计算梯度、Hessian和函数值
                f_x, g_x, H_x = func.value_gradient_hessian(x)
                
                # 检查收敛
                if self._check_convergence(g_x, i):
                    message = f"Converged at iteration {i}"
                    break
                
                # 确保Hessian正定
                if self.modify_hessian:
                    H_x = modify_to_positive_definite(H_x)
                
                # 求解牛顿方程: H * p = -g
                p = solve(H_x, -g_x, assume_a='sym')
                
                # 线搜索确定步长
                if self.line_search:
                    alpha = self._backtracking_line_search(func, x, p, f_x, g_x)
                    step = alpha * p
                else:
                    # 标准牛顿法：步长为1
                    step = p
                
                # 更新迭代点
                x_new = x + step
                self._store_history(x_new)
                
                # 执行回调和打印
                self._execute_callback(i, x_new, func.value(x_new), func.gradient(x_new))
                self._print_iteration(i, func.value(x_new), np.linalg.norm(g_x))
                x = x_new
                
            except np.linalg.LinAlgError:
                message = "Hessian matrix is singular"
                if self.verbose:
                    print(message)
                break
            except Exception as e:
                message = f"Optimization failed: {str(e)}"
                if self.verbose:
                    print(message)
                break
        else:
            message = "Maximum iterations reached"
            if self.verbose:
                print(message)
        
        final_grad = func.gradient(x)
        success = np.linalg.norm(final_grad) < self.tol
        
        return OptimizationResult(
            x=x,
            fun=func.value(x),
            niter=i,
            success=success,
            message=message,
            history=np.array(self.history),
            grad_norm=np.linalg.norm(final_grad)
        )
    
    def _backtracking_line_search(self, func, x, p, f_x, g_x, 
                                 alpha0: float = 1.0, rho: float = 0.5, 
                                 c: float = 1e-4) -> float:
        alpha = alpha0
        directional_derivative = np.dot(g_x, p)
        
        # Armijo: f(x+αp)<=f(x)+c*α*(g^T p)
        while func.value(x + alpha * p) > f_x + c * alpha * directional_derivative:
            alpha *= rho
            if alpha < 1e-10:
                break
        return alpha