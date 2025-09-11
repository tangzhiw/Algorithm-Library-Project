import numpy as np
from ..core.base_optimizer import BaseOptimizer
from ..core.line_search import backtracking_line_search


class NewtonMethod(BaseOptimizer):
	"""
	牛顿法优化器
	"""
	
	def __init__(self, line_search=True, **kwargs):
		"""
		初始化牛顿法优化器

		参数:
			line_search: 是否使用线搜索
			**kwargs: 基类参数
		"""
		super().__init__(**kwargs)
		self.line_search = line_search
	
	def optimize(self, f, x0, grad_f=None, hessian_f=None):
		"""
		执行牛顿法优化

		参数:
			f: 目标函数
			x0: 初始点
			grad_f: 梯度函数
			hessian_f: Hessian矩阵函数

		返回:
			优化结果字典
		"""
		if grad_f is None or hessian_f is None:
			raise ValueError("牛顿法需要梯度函数和Hessian矩阵函数")
		
		x = x0.copy()
		self.reset_history()
		
		for i in range(self.max_iter):
			# 计算梯度、Hessian和函数值
			grad = grad_f(x)
			hessian = hessian_f(x)
			f_val = f(x)
			grad_norm = np.linalg.norm(grad)
			
			# 记录历史
			self._record_history(x, f_val, grad_norm)
			
			# 检查收敛
			if grad_norm < self.tol:
				break
			
			try:
				# 计算牛顿方向
				direction = -np.linalg.solve(hessian, grad)
			except np.linalg.LinAlgError:
				# 如果Hessian矩阵不可逆，使用梯度下降方向
				direction = -grad
			
			# 确定步长
			if self.line_search:
				alpha = backtracking_line_search(f, x, direction, grad)
			else:
				alpha = 1.0
			
			# 更新参数
			x = x + alpha * direction
		
		return self.get_result()