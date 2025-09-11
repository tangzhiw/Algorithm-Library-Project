# 正确的导入方式
from OptimizationLibrary.optimizers import GradientDescent, NewtonMethod, BFGS, DFP, ConjugateGradient
from OptimizationLibrary.test_functions import simple_quadratic, rosenbrock_function
from OptimizationLibrary.utils import plot_optimization_2d, plot_optimization_3d, plot_convergence
from OptimizationLibrary.benchmarks import run_quadratic_benchmark, run_rosenbrock_benchmark
import numpy as np

def example_gradient_descent():
	"""梯度下降法示例"""
	print("=" * 50)
	print("梯度下降法示例")
	print("=" * 50)
	
	# 创建测试函数
	f, grad_f, _ = simple_quadratic()
	x0 = np.array([100.0, 100.0])
	
	# 创建优化器
	optimizer = GradientDescent(learning_rate=0.1, max_iter=100, tol=1e-6)
	
	# 运行优化
	result = optimizer.optimize(f, x0, grad_f)
	
	# 打印结果
	print(f"初始点: {x0}")
	print(f"最优解: {result['x']}")
	print(f"最优值: {result['f']}")
	print(f"迭代次数: {result['iterations']}")
	print(f"是否收敛: {result['converged']}")
	
	# 可视化
	plot_optimization_2d(f, result['history']['x'], "梯度下降法优化过程")
	plot_convergence(result['history']['f'], "梯度下降法收敛过程")
	
	return result


def example_newton_method():
	"""牛顿法示例"""
	print("=" * 50)
	print("牛顿法示例")
	print("=" * 50)
	
	# 创建测试函数
	f, grad_f, hessian_f = simple_quadratic()
	x0 = np.array([100.0, 100.0])
	
	# 创建优化器
	optimizer = NewtonMethod(max_iter=100, tol=1e-6)
	
	# 运行优化
	result = optimizer.optimize(f, x0, grad_f, hessian_f)
	
	# 打印结果
	print(f"初始点: {x0}")
	print(f"最优解: {result['x']}")
	print(f"最优值: {result['f']}")
	print(f"迭代次数: {result['iterations']}")
	print(f"是否收敛: {result['converged']}")
	
	# 可视化
	plot_optimization_2d(f, result['history']['x'], "牛顿法优化过程")
	plot_convergence(result['history']['f'], "牛顿法收敛过程")
	
	return result


def example_bfgs():
	"""BFGS拟牛顿法示例"""
	print("=" * 50)
	print("BFGS拟牛顿法示例")
	print("=" * 50)
	
	# 创建测试函数
	f, grad_f, _ = simple_quadratic()
	x0 = np.array([100.0, 100.0])
	
	# 创建优化器
	optimizer = BFGS(max_iter=100, tol=1e-6)
	
	# 运行优化
	result = optimizer.optimize(f, x0, grad_f)
	
	# 打印结果
	print(f"初始点: {x0}")
	print(f"最优解: {result['x']}")
	print(f"最优值: {result['f']}")
	print(f"迭代次数: {result['iterations']}")
	print(f"是否收敛: {result['converged']}")
	
	# 可视化
	plot_optimization_2d(f, result['history']['x'], "BFGS优化过程")
	plot_convergence(result['history']['f'], "BFGS收敛过程")
	
	return result


def example_dfp():
	"""DFP拟牛顿法示例"""
	print("=" * 50)
	print("DFP拟牛顿法示例")
	print("=" * 50)
	
	# 创建测试函数
	f, grad_f, _ = simple_quadratic()
	x0 = np.array([100.0, 100.0])
	
	# 创建优化器
	optimizer = DFP(max_iter=100, tol=1e-6)
	
	# 运行优化
	result = optimizer.optimize(f, x0, grad_f)
	
	# 打印结果
	print(f"初始点: {x0}")
	print(f"最优解: {result['x']}")
	print(f"最优值: {result['f']}")
	print(f"迭代次数: {result['iterations']}")
	print(f"是否收敛: {result['converged']}")
	
	# 可视化
	plot_optimization_2d(f, result['history']['x'], "DFP优化过程")
	plot_convergence(result['history']['f'], "DFP收敛过程")
	
	return result


def example_conjugate_gradient():
	"""共轭梯度法示例"""
	print("=" * 50)
	print("共轭梯度法示例")
	print("=" * 50)
	
	# 创建测试函数
	f, grad_f, _ = simple_quadratic()
	x0 = np.array([100.0, 100.0])
	
	# 创建优化器
	optimizer = ConjugateGradient(method='fletcher_reeves', max_iter=100, tol=1e-6)
	
	# 运行优化
	result = optimizer.optimize(f, x0, grad_f)
	
	# 打印结果
	print(f"初始点: {x0}")
	print(f"最优解: {result['x']}")
	print(f"最优值: {result['f']}")
	print(f"迭代次数: {result['iterations']}")
	print(f"是否收敛: {result['converged']}")
	
	# 可视化
	plot_optimization_2d(f, result['history']['x'], "共轭梯度法优化过程")
	plot_convergence(result['history']['f'], "共轭梯度法收敛过程")
	
	return result


def example_rosenbrock():
	"""Rosenbrock函数优化示例"""
	print("=" * 50)
	print("Rosenbrock函数优化示例")
	print("=" * 50)
	
	# 创建测试函数
	f, grad_f, hessian_f = rosenbrock_function()
	x0 = np.array([-1.5, 1.5])
	
	# 创建优化器
	optimizer = BFGS(max_iter=1000, tol=1e-6)
	
	# 运行优化
	result = optimizer.optimize(f, x0, grad_f)
	
	# 打印结果
	print(f"初始点: {x0}")
	print(f"最优解: {result['x']}")
	print(f"最优值: {result['f']}")
	print(f"迭代次数: {result['iterations']}")
	print(f"是否收敛: {result['converged']}")
	
	# 可视化
	plot_optimization_2d(f, result['history']['x'], "Rosenbrock函数BFGS优化过程", x_lim=(-2, 2), y_lim=(-1, 3))
	plot_convergence(result['history']['f'], "Rosenbrock函数BFGS收敛过程")
	
	return result


if __name__ == "__main__":
	# 运行各个示例
	gd_result = example_gradient_descent()
	newton_result = example_newton_method()
	bfgs_result = example_bfgs()
	dfp_result = example_dfp()
	cg_result = example_conjugate_gradient()
	rosenbrock_result = example_rosenbrock()
	
	# 运行性能测试
	print("\n" + "=" * 50)
	print("二次函数性能测试")
	print("=" * 50)
	quadratic_benchmark = run_quadratic_benchmark()
	
	print("\n" + "=" * 50)
	print("Rosenbrock函数性能测试")
	print("=" * 50)
	rosenbrock_benchmark = run_rosenbrock_benchmark()