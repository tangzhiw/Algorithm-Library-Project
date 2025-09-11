import numpy as np


def backtracking_line_search(f, x, direction, grad, alpha=1.0, beta=0.5, sigma=0.1):
	"""
	回溯线搜索算法

	参数:
		f: 目标函数
		x: 当前点
		direction: 搜索方向
		grad: 当前梯度
		alpha: 初始步长
		beta: 步长衰减因子
		sigma: Armijo条件参数

	返回:
		最优步长
	"""
	# 确保方向是下降方向
	if np.dot(grad, direction) >= 0:
		return 0.0
	
	# Armijo条件
	f_current = f(x)
	while f(x + alpha * direction) > f_current + sigma * alpha * np.dot(grad, direction):
		alpha *= beta
		
		# 防止步长过小
		if alpha < 1e-10:
			break
	
	return alpha


def wolfe_line_search(f, grad_f, x, direction, alpha=1.0, beta=0.5, c1=1e-4, c2=0.9, max_iter=20):
	"""
	Wolfe条件线搜索

	参数:
		f: 目标函数
		grad_f: 梯度函数
		x: 当前点
		direction: 搜索方向
		alpha: 初始步长
		beta: 步长衰减因子
		c1: Armijo条件参数
		c2: 曲率条件参数
		max_iter: 最大迭代次数

	返回:
		最优步长
	"""
	# 确保方向是下降方向
	grad = grad_f(x)
	if np.dot(grad, direction) >= 0:
		return 0.0
	
	f_current = f(x)
	grad_current = grad
	
	for _ in range(max_iter):
		x_new = x + alpha * direction
		f_new = f(x_new)
		grad_new = grad_f(x_new)
		
		# Armijo条件
		armijo_condition = f_new <= f_current + c1 * alpha * np.dot(grad_current, direction)
		
		# 曲率条件
		curvature_condition = np.dot(grad_new, direction) >= c2 * np.dot(grad_current, direction)
		
		if armijo_condition and curvature_condition:
			return alpha
		
		alpha *= beta
	
	return alpha