import numpy as np
from ..core.base_optimizer import BaseOptimizer
from ..core.line_search import wolfe_line_search


class ConjugateGradient(BaseOptimizer):
	"""
	共轭梯度法优化器
	"""
	
	def __init__(self, method='fletcher_reeves', restart_iter=None, **kwargs):
		"""
		初始化共轭梯度优化器

		参数:
			method: 共轭梯度方法 ('fletcher_reeves', 'polak_ribiere', 'hestenes_stiefel')
			restart_iter: 重启迭代次数（None表示不重启）
			**kwargs: 基类参数
		"""
		super().__init__(**kwargs)
		self.method = method
		self.restart_iter = restart_iter
	
	def optimize(self, f, x0, grad_f=None):
		"""
		执行共轭梯度优化

		参数:
			f: 目标函数
			x0: 初始点
			grad_f: 梯度函数

		返回:
			优化结果字典
		"""
		if grad_f is None:
			raise ValueError("共轭梯度法需要梯度函数")
		
		x = x0.copy()
		self.reset_history()
		
		# 初始梯度
		grad = grad_f(x)
		direction = -grad  # 初始方向为负梯度
		
		for i in range(self.max_iter):
			# 计算函数值和梯度范数
			f_val = f(x)
			grad_norm = np.linalg.norm(grad)
			
			# 记录历史
			self._record_history(x, f_val, grad_norm)
			
			# 检查收敛
			if grad_norm < self.tol:
				break
			
			# 确定步长
			alpha = wolfe_line_search(f, grad_f, x, direction)
			
			# 更新参数
			x_new = x + alpha * direction
			
			# 计算新梯度
			grad_new = grad_f(x_new)
			
			# 检查是否需要重启
			if self.restart_iter and (i + 1) % self.restart_iter == 0:
				beta = 0
			else:
				# 计算beta值
				if self.method == 'fletcher_reeves':
					beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
				elif self.method == 'polak_ribiere':
					beta = np.dot(grad_new, grad_new - grad) / np.dot(grad, grad)
				elif self.method == 'hestenes_stiefel':
					y = grad_new - grad
					beta = np.dot(grad_new, y) / np.dot(direction, y)
				else:
					raise ValueError(f"未知的共轭梯度方法: {self.method}")
			
			# 更新搜索方向
			direction = -grad_new + beta * direction
			
			# 确保方向是下降方向
			if np.dot(direction, grad_new) >= 0:
				direction = -grad_new
			
			# 更新参数和梯度
			x = x_new
			grad = grad_new
		
		return self.get_result()