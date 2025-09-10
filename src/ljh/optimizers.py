import numpy as np

class Optimizers:
    """三种优化算法的简化实现"""
    
    def __init__(self):
        self.optimizer_names = ["AdaGrad", "RMSProp", "Adam"]
    
    def adagrad(self, x, y, dx, dy, learning_rate, cache_x, cache_y, epsilon=1e-8):
        """AdaGrad优化算法"""
        cache_x += dx ** 2
        cache_y += dy ** 2
        
        x -= learning_rate * dx / (np.sqrt(cache_x) + epsilon)
        y -= learning_rate * dy / (np.sqrt(cache_y) + epsilon)
        
        return x, y, cache_x, cache_y
    
    def rmsprop(self, x, y, dx, dy, learning_rate, cache_x, cache_y, beta=0.9, epsilon=1e-8):
        """RMSProp优化算法"""
        cache_x = beta * cache_x + (1 - beta) * dx ** 2
        cache_y = beta * cache_y + (1 - beta) * dy ** 2
        
        x -= learning_rate * dx / (np.sqrt(cache_x) + epsilon)
        y -= learning_rate * dy / (np.sqrt(cache_y) + epsilon)
        
        return x, y, cache_x, cache_y
    
    def adam(self, x, y, dx, dy, learning_rate, m_x, m_y, v_x, v_y, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam优化算法"""
        t += 1
        
        # 更新一阶矩估计
        m_x = beta1 * m_x + (1 - beta1) * dx
        m_y = beta1 * m_y + (1 - beta1) * dy
        
        # 更新二阶矩估计
        v_x = beta2 * v_x + (1 - beta2) * (dx ** 2)
        v_y = beta2 * v_y + (1 - beta2) * (dy ** 2)
        
        # 偏差校正
        m_x_hat = m_x / (1 - beta1 ** t)
        m_y_hat = m_y / (1 - beta1 ** t)
        v_x_hat = v_x / (1 - beta2 ** t)
        v_y_hat = v_y / (1 - beta2 ** t)
        
        # 更新参数
        x -= learning_rate * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        y -= learning_rate * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
        
        return x, y, m_x, m_y, v_x, v_y, t