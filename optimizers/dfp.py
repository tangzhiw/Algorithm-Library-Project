import numpy as np
from ..core.base_optimizer import BaseOptimizer
from ..core.line_search import wolfe_line_search


class DFP(BaseOptimizer):
	"""
	DFP拟牛顿法优化器
	"""
	
	def __init__(self, line_search=True, **kwargs):
		"""
		初始化DFP优化器

		参数:
			line_search: 是否使用线搜索
			**kwargs: 基类参数
		"""
		super().__init__(**kwargs)
		self.line_search = line_search
	
	def optimize(self, f, x0, grad_f=None):
		"""
		执行DFP优化

		参数:
			f: 目标函数
			x0: 初始点
			grad_f: 梯度函数

		返回:
			优化结果字典
		"""
		if grad_f is None:
			raise ValueError("DFP法需要梯度函数")
		
		n = len(x0)
		x = x0.copy()
		H = np.eye(n)  # 初始近似Hessian逆矩阵
		self.reset_history()
		
		# 初始梯度
		grad = grad_f(x)
		
		for i in range(self.max_iter):
			# 计算函数值和梯度范数
			f_val = f(x)
			grad_norm = np.linalg.norm(grad)
			
			# 记录历史
			self._record_history(x, f_val, grad_norm)
			
			# 检查收敛
			if grad_norm < self.tol:
				break
			
			# 计算搜索方向
			direction = -np.dot(H, grad)
			
			# 确定步长
			if self.line_search:
				alpha = wolfe_line_search(f, grad_f, x, direction)
			else:
				alpha = 1.0
			
			# 更新参数
			x_new = x + alpha * direction
			
			# 计算新梯度
			grad_new = grad_f(x_new)
			
			# 计算梯度变化和参数变化
			s = x_new - x
			y = grad_new - grad
			
			# 检查分母是否为零（避免数值问题）
			if np.dot(y, s) < 1e-10:
				break
			
			# 更新Hessian逆矩阵的近似 (DFP公式)
			rho = 1.0 / np.dot(y, s)
			H = H - np.dot(np.dot(H, np.outer(y, y)), H) / np.dot(y, np.dot(H, y)) + rho * np.outer(s, s)
			
			# 更新参数和梯度
			x = x_new
			grad = grad_new
		
		return self.get_result()