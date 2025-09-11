import numpy as np
from ..core.base_optimizer import BaseOptimizer
from ..core.line_search import backtracking_line_search


class GradientDescent(BaseOptimizer):
	"""
	梯度下降法优化器
	"""
	
	def __init__(self, learning_rate=0.01, line_search=False, **kwargs):
		"""
		初始化梯度下降优化器

		参数:
			learning_rate: 学习率（固定步长时使用）
			line_search: 是否使用线搜索
			**kwargs: 基类参数
		"""
		super().__init__(**kwargs)
		self.learning_rate = learning_rate
		self.line_search = line_search
	
	def optimize(self, f, x0, grad_f=None):
		"""
		执行梯度下降优化

		参数:
			f: 目标函数
			x0: 初始点
			grad_f: 梯度函数

		返回:
			优化结果字典
		"""
		if grad_f is None:
			raise ValueError("梯度下降法需要梯度函数")
		
		x = x0.copy()
		self.reset_history()
		
		for i in range(self.max_iter):
			# 计算梯度和函数值
			grad = grad_f(x)
			f_val = f(x)
			grad_norm = np.linalg.norm(grad)
			
			# 记录历史
			self._record_history(x, f_val, grad_norm)
			
			# 检查收敛
			if grad_norm < self.tol:
				break
			
			# 确定步长
			if self.line_search:
				# 使用线搜索确定步长
				direction = -grad
				alpha = backtracking_line_search(f, x, direction, grad)
			else:
				# 使用固定学习率
				alpha = self.learning_rate
			
			# 更新参数
			x = x - alpha * grad
		
		return self.get_result()