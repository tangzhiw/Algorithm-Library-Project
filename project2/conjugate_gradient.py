import numpy as np
from base import Optimizer, OptimizationResult  
from line_search import strong_wolfe_line_search  

class ConjugateGradient(Optimizer):
    def __init__(self, tol: float = 1e-6, max_iter: int = 1000, 
                 verbose: bool = False, history: bool = True,
                 restart_iter: int = 50):
        super().__init__(tol, max_iter, verbose, history)
        self.restart_iter = restart_iter
    
    def minimize(self, func, x0):
        # 初始化
        x = np.asarray(x0, dtype=float).copy()
        self.history = []
        self._store_history(x)
        
        # 初始梯度和搜索方向
        g = func.gradient(x)
        p = -g  # 初始方向为负梯度方向
        
        # 主迭代循环
        for i in range(self.max_iter):
            # 检查收敛
            if self._check_convergence(g, i):
                break
            
            # 周期性重启（防止数值不稳定）
            if i % self.restart_iter == 0:
                p = -func.gradient(x)
            
            # 线搜索确定步长
            alpha = strong_wolfe_line_search(func, x, p)
            
            # 更新迭代点
            x_new = x + alpha * p
            self._store_history(x_new)
            
            # 计算新梯度
            g_new = func.gradient(x_new)
            
            # Fletcher-Reeves公式计算beta
            beta = np.dot(g_new, g_new) / np.dot(g, g)
            
            # 更新搜索方向
            p = -g_new + beta * p
            
            # 为下一次迭代更新变量
            x = x_new
            g = g_new
            
            self._print_iteration(i, func.value(x), np.linalg.norm(g))
        
        else:
            message = "Maximum iterations reached"
            if self.verbose:
                print(message)
        
        # 计算最终结果
        final_grad = func.gradient(x)
        success = np.linalg.norm(final_grad) < self.tol
        
        return OptimizationResult(
            x=x,
            fun=func.value(x),
            niter=i,
            success=success,
            message=message if 'message' in locals() else "Optimization completed",
            history=np.array(self.history),
            grad_norm=np.linalg.norm(final_grad)
        )