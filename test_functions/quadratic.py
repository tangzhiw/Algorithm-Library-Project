import numpy as np


def quadratic_function(A, b, c=0):
	"""
	生成二次函数 f(x) = 0.5 * x^T A x + b^T x + c

	参数:
		A: 二次项系数矩阵
		b: 一次项系数向量
		c: 常数项

	返回:
		目标函数、梯度函数和Hessian矩阵函数
	"""
	
	def f(x):
		return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x) + c
	
	def grad_f(x):
		return np.dot(A, x) + b
	
	def hessian_f(x):
		return A
	
	return f, grad_f, hessian_f


def simple_quadratic():
	"""
	简单的二次函数 f(x, y) = x^2 + y^2
	"""
	A = np.array([[2, 0], [0, 2]])
	b = np.array([0, 0])
	return quadratic_function(A, b)


def rosenbrock_function(a=1, b=100):
	"""
	Rosenbrock函数: f(x, y) = (a - x)^2 + b(y - x^2)^2
	"""
	
	def f(x):
		return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
	
	def grad_f(x):
		df_dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
		df_dy = 2 * b * (x[1] - x[0] ** 2)
		return np.array([df_dx, df_dy])
	
	def hessian_f(x):
		d2f_dx2 = 2 - 4 * b * (x[1] - 3 * x[0] ** 2)
		d2f_dxdy = -4 * b * x[0]
		d2f_dydx = -4 * b * x[0]
		d2f_dy2 = 2 * b
		return np.array([[d2f_dx2, d2f_dxdy], [d2f_dydx, d2f_dy2]])
	
	return f, grad_f, hessian_f