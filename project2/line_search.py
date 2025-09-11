import numpy as np
from typing import Callable
from base import ObjectiveFunction

def backtracking_line_search(func: ObjectiveFunction, x: np.ndarray, 
                            p: np.ndarray, alpha0: float = 1.0, 
                            rho: float = 0.5, c: float = 1e-4) -> float:
    alpha = alpha0
    fx = func.value(x)
    g = func.gradient(x)
    directional_derivative = np.dot(g, p)
    
    # Armijo: f(x+αp)<=f(x)+c*α*(g^T p)
    while func.value(x + alpha * p) > fx + c * alpha * directional_derivative:
        alpha *= rho
        if alpha < 1e-10:
            break
    return alpha

def strong_wolfe_line_search(func: ObjectiveFunction, x: np.ndarray, 
                            p: np.ndarray, alpha_max: float = 10.0,
                            c1: float = 1e-4, c2: float = 0.9) -> float:
    alpha = 0.0
    beta = alpha_max
    alpha_prev = 0.0
    alpha_current = 1.0
    
    fx = func.value(x)
    g = func.gradient(x)
    directional_derivative = np.dot(g, p)
    
    for _ in range(20):  
        # 计算当前点的函数值和梯度
        x_current = x + alpha_current * p
        f_current = func.value(x_current)
        g_current = func.gradient(x_current)
        directional_derivative_current = np.dot(g_current, p)
        
        # 检查Armijo条件
        if f_current > fx + c1 * alpha_current * directional_derivative:
            beta = alpha_current
            alpha_current = (alpha + beta) / 2
        # 检查曲率条件
        elif abs(directional_derivative_current) > -c2 * directional_derivative:
            alpha = alpha_current
            alpha_current = (alpha + beta) / 2
        else:
            return alpha_current
        
        # 防止无限循环
        if abs(alpha_current - alpha_prev) < 1e-10:
            break
        alpha_prev = alpha_current
    
    return alpha_current