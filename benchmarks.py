import numpy as np
import time
import matplotlib.pyplot as plt
from .optimizers import GradientDescent, NewtonMethod, BFGS, DFP, ConjugateGradient
from .test_functions.quadratic import simple_quadratic, rosenbrock_function
from .utils.visualization import plot_algorithm_comparison


class Benchmark:
	"""
	性能测试框架
	"""
	
	def __init__(self):
		self.results = {}
	
	def run_benchmark(self, f, grad_f, hessian_f, x0, optimizers, max_iter=100, tol=1e-6):
		"""
		运行性能测试

		参数:
			f: 目标函数
			grad_f: 梯度函数
			hessian_f: Hessian矩阵函数
			x0: 初始点
			optimizers: 优化器列表
			max_iter: 最大迭代次数
			tol: 收敛容差
		"""
		self.results = {}
		
		for optimizer_name, optimizer_class in optimizers:
			print(f"运行 {optimizer_name}...")
			
			# 创建优化器实例
			if optimizer_name in ['牛顿法']:
				opt = optimizer_class(max_iter=max_iter, tol=tol)
			else:
				opt = optimizer_class(max_iter=max_iter, tol=tol)
			
			# 运行优化
			start_time = time.time()
			
			if optimizer_name in ['牛顿法']:
				result = opt.optimize(f, x0, grad_f, hessian_f)
			else:
				result = opt.optimize(f, x0, grad_f)
			
			end_time = time.time()
			
			# 记录结果
			self.results[optimizer_name] = {
				'time': end_time - start_time,
				'iterations': result['iterations'],
				'final_value': result['f'],
				'final_grad_norm': result['grad_norm'],
				'converged': result['converged'],
				'history': result['history']
			}
	
	def plot_results(self, title="优化算法性能比较"):
		"""
		绘制性能比较结果
		"""
		if not self.results:
			print("没有可用的结果，请先运行性能测试")
			return
		
		plot_algorithm_comparison(self.results, title)
	
	def print_results(self):
		"""
		打印性能测试结果
		"""
		if not self.results:
			print("没有可用的结果，请先运行性能测试")
			return
		
		print("优化算法性能测试结果:")
		print("=" * 80)
		for name, result in self.results.items():
			print(f"{name}:")
			print(f"  运行时间: {result['time']:.6f} 秒")
			print(f"  迭代次数: {result['iterations']}")
			print(f"  最终函数值: {result['final_value']:.6f}")
			print(f"  最终梯度范数: {result['final_grad_norm']:.6f}")
			print(f"  是否收敛: {result['converged']}")
			print()


# 示例性能测试
def run_quadratic_benchmark():
	"""运行二次函数的性能测试"""
	# 创建测试函数
	f, grad_f, hessian_f = simple_quadratic()
	x0 = np.array([100.0, 100.0])
	
	# 定义优化器
	optimizers = [
		('梯度下降法', GradientDescent),
		('牛顿法', NewtonMethod),
		('BFGS', BFGS),
		('DFP', DFP),
		('共轭梯度法', ConjugateGradient)
	]
	
	# 运行性能测试
	benchmark = Benchmark()
	benchmark.run_benchmark(f, grad_f, hessian_f, x0, optimizers, max_iter=100, tol=1e-6)
	benchmark.plot_results("二次函数性能比较")
	benchmark.print_results()
	
	return benchmark


def run_rosenbrock_benchmark():
	"""运行Rosenbrock函数的性能测试"""
	# 创建测试函数
	f, grad_f, hessian_f = rosenbrock_function()
	x0 = np.array([-1.5, 1.5])
	
	# 定义优化器
	optimizers = [
		('梯度下降法', GradientDescent),
		('牛顿法', NewtonMethod),
		('BFGS', BFGS),
		('DFP', DFP),
		('共轭梯度法', ConjugateGradient)
	]
	
	# 运行性能测试
	benchmark = Benchmark()
	benchmark.run_benchmark(f, grad_f, hessian_f, x0, optimizers, max_iter=1000, tol=1e-6)
	benchmark.plot_results("Rosenbrock函数性能比较")
	benchmark.print_results()
	
	return benchmark