import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_optimization_2d(f, x_history, title="优化过程", x_lim=None, y_lim=None):
	"""
	绘制2D优化过程

	参数:
		f: 目标函数
		x_history: 优化过程中的x值历史
		title: 图表标题
		x_lim: x轴范围 (None表示自动调整)
		y_lim: y轴范围 (None表示自动调整)
	"""
	# 提取路径坐标
	path_x = [p[0] for p in x_history]
	path_y = [p[1] for p in x_history]
	
	# 自动确定坐标轴范围
	if x_lim is None:
		x_min, x_max = min(path_x), max(path_x)
		x_range = x_max - x_min
		x_lim = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
	
	if y_lim is None:
		y_min, y_max = min(path_y), max(path_y)
		y_range = y_max - y_min
		y_lim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
	
	# 生成网格数据
	x = np.linspace(x_lim[0], x_lim[1], 100)
	y = np.linspace(y_lim[0], y_lim[1], 100)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)
	
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
	
	# 创建图形
	plt.figure(figsize=(10, 8))
	
	# 绘制等高线
	contour = plt.contour(X, Y, Z, 50, cmap=cm.viridis)
	plt.clabel(contour, fontsize=8)
	
	# 绘制优化路径
	plt.plot(path_x, path_y, 'ro-', linewidth=2, markersize=4)
	plt.plot(path_x[0], path_y[0], 'go', markersize=8, label='起点')
	plt.plot(path_x[-1], path_y[-1], 'bo', markersize=8, label='终点')
	
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.colorbar(contour)
	plt.show()


def plot_optimization_3d(f, x_history, title="优化过程", x_lim=None, y_lim=None):
	"""
	绘制3D优化过程

	参数:
		f: 目标函数
		x_history: 优化过程中的x值历史
		title: 图表标题
		x_lim: x轴范围 (None表示自动调整)
		y_lim: y轴范围 (None表示自动调整)
	"""
	# 提取路径坐标
	path_x = [p[0] for p in x_history]
	path_y = [p[1] for p in x_history]
	path_z = [f(p) for p in x_history]
	
	# 自动确定坐标轴范围
	if x_lim is None:
		x_min, x_max = min(path_x), max(path_x)
		x_range = x_max - x_min
		x_lim = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
	
	if y_lim is None:
		y_min, y_max = min(path_y), max(path_y)
		y_range = y_max - y_min
		y_lim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
	
	# 生成网格数据
	x = np.linspace(x_lim[0], x_lim[1], 100)
	y = np.linspace(y_lim[0], y_lim[1], 100)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)
	
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
	
	# 创建图形
	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111, projection='3d')
	
	# 绘制表面
	surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)
	
	# 绘制优化路径
	ax.plot(path_x, path_y, path_z, 'ro-', linewidth=2, markersize=4)
	ax.plot([path_x[0]], [path_y[0]], [path_z[0]], 'go', markersize=8, label='起点')
	ax.plot([path_x[-1]], [path_y[-1]], [path_z[-1]], 'bo', markersize=8, label='终点')
	
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('f(x, y)')
	ax.set_title(title)
	ax.legend()
	
	plt.colorbar(surf)
	plt.show()


def plot_interactive_optimization(f, x_history, title="优化过程", x_lim=None, y_lim=None):
	"""
	创建交互式优化过程可视化 (使用Plotly)

	参数:
		f: 目标函数
		x_history: 优化过程中的x值历史
		title: 图表标题
		x_lim: x轴范围 (None表示自动调整)
		y_lim: y轴范围 (None表示自动调整)
	"""
	# 提取路径坐标
	path_x = [p[0] for p in x_history]
	path_y = [p[1] for p in x_history]
	path_z = [f(p) for p in x_history]
	
	# 自动确定坐标轴范围
	if x_lim is None:
		x_min, x_max = min(path_x), max(path_x)
		x_range = x_max - x_min
		x_lim = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
	
	if y_lim is None:
		y_min, y_max = min(path_y), max(path_y)
		y_range = y_max - y_min
		y_lim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
	
	# 生成网格数据
	x = np.linspace(x_lim[0], x_lim[1], 100)
	y = np.linspace(y_lim[0], y_lim[1], 100)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)
	
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
	
	# 创建子图
	fig = make_subplots(
		rows=1, cols=2,
		subplot_titles=('等高线图与优化路径', '3D曲面与优化路径'),
		specs=[[{"type": "contour"}, {"type": "surface"}]]
	)
	
	# 添加等高线图
	fig.add_trace(
		go.Contour(
			x=x, y=y, z=Z,
			contours=dict(coloring='lines', showlabels=True),
			colorscale='Viridis'
		),
		row=1, col=1
	)
	
	# 添加等高线图上的路径
	fig.add_trace(
		go.Scatter(
			x=path_x, y=path_y,
			mode='lines+markers',
			marker=dict(color='red', size=6),
			line=dict(color='red', width=2),
			name='优化路径'
		),
		row=1, col=1
	)
	
	# 添加3D曲面
	fig.add_trace(
		go.Surface(
			x=x, y=y, z=Z,
			colorscale='Viridis',
			opacity=0.8,
			showscale=False
		),
		row=1, col=2
	)
	
	# 添加3D路径
	fig.add_trace(
		go.Scatter3d(
			x=path_x, y=path_y, z=path_z,
			mode='lines+markers',
			marker=dict(color='red', size=4),
			line=dict(color='red', width=3),
			name='优化路径'
		),
		row=1, col=2
	)
	
	# 更新布局
	fig.update_layout(
		title=title,
		width=1000,
		height=500
	)
	
	fig.show()


def plot_convergence(f_history, title="收敛过程"):
	"""
	绘制收敛过程

	参数:
		f_history: 函数值历史
		title: 图表标题
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(f_history, 'b-', linewidth=2)
	plt.xlabel('迭代次数')
	plt.ylabel('函数值')
	plt.title(title)
	plt.grid(True)
	plt.yscale('log')  # 使用对数尺度更好地显示收敛
	plt.show()


def plot_algorithm_comparison(results, title="算法性能比较"):
	"""
	绘制算法性能比较图

	参数:
		results: 包含算法性能结果的字典
		title: 图表标题
	"""
	names = list(results.keys())
	times = [results[name]['time'] for name in names]
	iterations = [results[name]['iterations'] for name in names]
	final_values = [results[name]['final_value'] for name in names]
	final_grad_norms = [results[name]['final_grad_norm'] for name in names]
	
	# 创建图形
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
	
	# 绘制运行时间
	ax1.bar(names, times)
	ax1.set_ylabel('时间 (秒)')
	ax1.set_title('运行时间比较')
	ax1.tick_params(axis='x', rotation=45)
	
	# 绘制迭代次数
	ax2.bar(names, iterations)
	ax2.set_ylabel('迭代次数')
	ax2.set_title('迭代次数比较')
	ax2.tick_params(axis='x', rotation=45)
	
	# 绘制最终函数值
	ax3.bar(names, final_values)
	ax3.set_ylabel('最终函数值')
	ax3.set_title('最终函数值比较')
	ax3.tick_params(axis='x', rotation=45)
	
	# 绘制最终梯度范数
	ax4.bar(names, final_grad_norms)
	ax4.set_ylabel('最终梯度范数')
	ax4.set_title('最终梯度范数比较')
	ax4.tick_params(axis='x', rotation=45)
	ax4.set_yscale('log')
	
	plt.suptitle(title)
	plt.tight_layout()
	plt.show()
	
	# 绘制收敛曲线
	plt.figure(figsize=(10, 6))
	for name in names:
		history = results[name]['history']
		plt.plot(history['f'], label=name, linewidth=2)
	
	plt.xlabel('迭代次数')
	plt.ylabel('函数值')
	plt.title('收敛速度比较')
	plt.legend()
	plt.grid(True)
	plt.yscale('log')
	plt.show()