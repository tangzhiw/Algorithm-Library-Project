import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_performance_table():
	"""创建算法性能对比表格"""
	# 算法名称
	algorithms = ['梯度下降法', '牛顿法', 'BFGS', 'DFP', '共轭梯度法']
	
	# 性能数据
	data = [
		[0.009647, 1000, 'NaN', 'NaN', '否'],
		[0.000000, 24, '0.000000', '0.000000', '是'],
		[0.008508, 5, '2.390383', '5.780892', '否'],
		[0.000000, 5, '2.390256', '5.745397', '否'],
		[0.011523, 108, '0.000000', '0.000001', '是']
	]
	
	# 创建图表
	fig, ax = plt.subplots(figsize=(14, 8))
	ax.axis('tight')
	ax.axis('off')
	
	# 创建表格
	table = ax.table(cellText=data,
	                 rowLabels=algorithms,
	                 colLabels=['运行时间(秒)', '迭代次数', '最终函数值', '最终梯度范数', '是否收敛'],
	                 loc='center',
	                 cellLoc='center',
	                 colWidths=[0.15, 0.15, 0.2, 0.2, 0.15])
	
	# 设置表格样式
	table.auto_set_font_size(False)
	table.set_fontsize(11)
	table.scale(1.2, 2.2)
	
	# 设置标题
	plt.title('优化算法性能对比表\n(测试函数: Rosenbrock函数, 初始点: [-1.0, 1.0])',
	          fontsize=16, fontweight='bold', pad=20)
	
	# 突出显示最佳性能
	# 运行时间列 - 找到最小值
	time_values = [0.009647, 0.000000, 0.008508, 0.000000, 0.011523]
	min_time = min(time_values)
	for i, val in enumerate(time_values):
		if val == min_time:
			table[(i + 1, 0)].set_facecolor('#90EE90')  # 浅绿色
	
	# 迭代次数列 - 找到最小值
	iter_values = [1000, 24, 5, 5, 108]
	min_iter = min(iter_values)
	for i, val in enumerate(iter_values):
		if val == min_iter:
			table[(i + 1, 1)].set_facecolor('#90EE90')  # 浅绿色
	
	# 最终函数值列 - 找到最小值
	func_values = [float('inf'), 0.000000, 2.390383, 2.390256, 0.000000]
	min_func = min(func_values)
	for i, val in enumerate(func_values):
		if val == min_func:
			table[(i + 1, 2)].set_facecolor('#90EE90')  # 浅绿色
	
	# 最终梯度范数列 - 找到最小值
	grad_values = [float('inf'), 0.000000, 5.780892, 5.745397, 0.000001]
	min_grad = min(grad_values)
	for i, val in enumerate(grad_values):
		if val == min_grad:
			table[(i + 1, 3)].set_facecolor('#90EE90')  # 浅绿色
	
	# 是否收敛列 - 标记"是"的单元格
	for i, row in enumerate(data):
		if row[4] == '是':
			table[(i + 1, 4)].set_facecolor('#90EE90')  # 浅绿色
	
	# 设置列标题样式
	for j in range(5):
		table[(0, j)].set_facecolor('#4F81BD')
		table[(0, j)].set_text_props(weight='bold', color='white')
	
	# 设置行标题样式
	for i in range(5):
		table[(i + 1, -1)].set_facecolor('#D9E1F2')
		table[(i + 1, -1)].set_text_props(weight='bold')
	
	# 添加性能分析
	analysis_text = (
		"性能分析:\n"
		"• 牛顿法表现最佳: 收敛最快且精度最高\n"
		"• 共轭梯度法也成功收敛，但需要更多迭代\n"
		"• BFGS和DFP收敛速度快但陷入局部最优\n"
		"• 梯度下降法未能收敛，达到最大迭代次数"
	)
	
	plt.figtext(0.1, 0.02, analysis_text,
	            fontsize=11, style='italic', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
	
	plt.tight_layout()
	plt.savefig('optimization_performance_comparison_v2.png', dpi=300, bbox_inches='tight')
	plt.show()


# 运行函数创建表格
create_performance_table()