import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
	"""
	优化器基类，定义所有优化算法的通用接口
	"""
	
	def __init__(self, max_iter=1000, tol=1e-6, callback=None):
		"""
		初始化优化器

		参数:
			max_iter: 最大迭代次数
			tol: 收敛容差
			callback: 每次迭代后的回调函数
		"""
		self.max_iter = max_iter
		self.tol = tol
		self.callback = callback
		self.history = {
			'x': [],
			'f': [],
			'grad_norm': [],
			'iterations': 0
		}
	
	@abstractmethod
	def optimize(self, f, x0, grad_f=None):
		"""
		执行优化过程

		参数:
			f: 目标函数
			x0: 初始点
			grad_f: 梯度函数（可选）

		返回:
			优化结果字典
		"""
		pass
	
	def _record_history(self, x, f_val, grad_norm):
		"""记录优化历史"""
		self.history['x'].append(x.copy())
		self.history['f'].append(f_val)
		self.history['grad_norm'].append(grad_norm)
		self.history['iterations'] += 1
		
		if self.callback:
			self.callback(x, f_val, grad_norm, self.history['iterations'])
	
	def reset_history(self):
		"""重置历史记录"""
		self.history = {
			'x': [],
			'f': [],
			'grad_norm': [],
			'iterations': 0
		}
	
	def get_result(self):
		"""获取优化结果"""
		if not self.history['x']:
			return None
		
		return {
			'x': self.history['x'][-1],
			'f': self.history['f'][-1],
			'grad_norm': self.history['grad_norm'][-1],
			'iterations': self.history['iterations'],
			'converged': self.history['grad_norm'][-1] < self.tol,
			'history': self.history
		}