import numpy as np

class TestFunctions:
    """简单的平方和损失函数"""
    
    def __init__(self):
        self.function_name = "x² + y²"
    
    def compute_loss(self, x, y):
        """计算损失函数值: f(x,y) = x² + y²"""
        return x**2 + y**2
    
    def compute_gradients(self, x, y):
        """计算梯度"""
        dx = 2 * x  # ∂f/∂x = 2x
        dy = 2 * y  # ∂f/∂y = 2y
        return dx, dy
    
    def create_meshgrid(self, range_val=(-5, 5), n_points=100):
        """创建网格用于可视化"""
        x = np.linspace(range_val[0], range_val[1], n_points)
        y = np.linspace(range_val[0], range_val[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = self.compute_loss(X, Y)
        return X, Y, Z