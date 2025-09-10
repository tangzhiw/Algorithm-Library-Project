import numpy as np
import matplotlib.pyplot as plt
from optimizers import Optimizers
from test_functions import TestFunctions
import inspect

def run_optimization_comparison():
    """运行三种优化算法的比较"""
    
    # 初始化
    optimizers = Optimizers()
    test_function = TestFunctions()
    
    # 公共参数
    initial_x, initial_y = 10.0, 10.0  # 初始点
    learning_rate = 0.5
    iterations = 120
    
    # 存储每种算法的优化路径
    paths = {
        "AdaGrad": [],
        "RMSProp": [], 
        "Adam": []
    }
    
    # 运行AdaGrad
    print("Running AdaGrad...")
    x, y = initial_x, initial_y
    cache_x, cache_y = 0.0, 0.0
    paths["AdaGrad"].append((x, y))
    
    for i in range(iterations):
        dx, dy = test_function.compute_gradients(x, y)
        x, y, cache_x, cache_y = optimizers.adagrad(x, y, dx, dy, learning_rate, cache_x, cache_y)
        paths["AdaGrad"].append((x, y))
    
    # 运行RMSProp
    print("Running RMSProp...")
    x, y = initial_x, initial_y
    cache_x, cache_y = 0.0, 0.0
    paths["RMSProp"].append((x, y))
    
    for i in range(iterations):
        dx, dy = test_function.compute_gradients(x, y)
        x, y, cache_x, cache_y = optimizers.rmsprop(x, y, dx, dy, learning_rate, cache_x, cache_y)
        paths["RMSProp"].append((x, y))
    
    # 运行Adam
    print("Running Adam...")
    x, y = initial_x, initial_y
    m_x, m_y, v_x, v_y = 0.0, 0.0, 0.0, 0.0
    t = 0
    paths["Adam"].append((x, y))
    
    for i in range(iterations):
        dx, dy = test_function.compute_gradients(x, y)
        x, y, m_x, m_y, v_x, v_y, t = optimizers.adam(x, y, dx, dy, learning_rate, m_x, m_y, v_x, v_y, t)
        paths["Adam"].append((x, y))
    
    # 打印最终结果
    print("\nFinal Results:")
    for optimizer, path in paths.items():
        final_x, final_y = path[-1]
        final_loss = test_function.compute_loss(final_x, final_y)
        print(f"{optimizer}: x={final_x:.6f}, y={final_y:.6f}, loss={final_loss:.6f}")
    
    return paths

def plot_optimization_paths(paths):
    """绘制优化路径的静态图"""
    
    test_function = TestFunctions()
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：等高线图上的优化路径
    X, Y, Z = test_function.create_meshgrid()
    
    # 绘制等高线
    contour = ax1.contour(X, Y, Z, 20, colors='gray', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # 绘制每种算法的路径
    colors = {'AdaGrad': 'blue', 'RMSProp': 'green', 'Adam': 'red'}
    markers = {'AdaGrad': 'o', 'RMSProp': 's', 'Adam': '^'}
    
    for optimizer, path in paths.items():
        path_array = np.array(path)
        ax1.plot(path_array[:, 0], path_array[:, 1], 
                color=colors[optimizer], linewidth=2, 
                label=f'{optimizer}', marker=markers[optimizer], 
                markersize=4, markevery=5)
    
    # 标记起点和终点
    ax1.plot(paths["AdaGrad"][0][0], paths["AdaGrad"][0][1], 
             'ko', markersize=8, label='Start')
    ax1.plot(0, 0, 'r*', markersize=12, label='Optimum (0,0)')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Optimization Paths on Contour Plot\nf(x,y) = x² + y²')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    # 右图：损失函数下降曲线
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Convergence Comparison')
    ax2.grid(True, alpha=0.3)
    
    for optimizer, path in paths.items():
        losses = [test_function.compute_loss(x, y) for x, y in path]
        ax2.plot(losses, label=optimizer, linewidth=2)
    
    ax2.legend()
    ax2.set_yscale('log')  # 使用对数坐标更好地显示收敛
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_dynamics(paths):
    """绘制学习动态的额外分析图"""
    
    test_function = TestFunctions()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. x坐标的变化
    for optimizer, path in paths.items():
        x_values = [point[0] for point in path]
        axes[0].plot(x_values, label=optimizer, linewidth=2)
    axes[0].set_title('x-coordinate Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('x value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. y坐标的变化
    for optimizer, path in paths.items():
        y_values = [point[1] for point in path]
        axes[1].plot(y_values, label=optimizer, linewidth=2)
    axes[1].set_title('y-coordinate Convergence')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('y value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 到最优点的距离
    for optimizer, path in paths.items():
        distances = [np.sqrt(x**2 + y**2) for x, y in path]
        axes[2].plot(distances, label=optimizer, linewidth=2)
    axes[2].set_title('Distance to Optimum (0,0)')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Distance')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. 最终损失值比较
    final_losses = []
    optimizers = []
    for optimizer, path in paths.items():
        final_x, final_y = path[-1]
        final_loss = test_function.compute_loss(final_x, final_y)
        final_losses.append(final_loss)
        optimizers.append(optimizer)
    
    bars = axes[3].bar(optimizers, final_losses, 
                      color=['blue', 'orange', 'green'], alpha=0.7)
    axes[3].set_title('Final Loss Values')
    axes[3].set_ylabel('Loss')
    axes[3].set_yscale('log')
    
    # 在柱状图上添加数值标签
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    savaPath = inspect.getfile(plot_learning_dynamics) + '/../优化算法收敛图.png'
    print(savaPath)
    plt.savefig(savaPath, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 运行优化比较
    paths = run_optimization_comparison()
    
    # 绘制详细分析图
    plot_learning_dynamics(paths)
    
    print("Visualization completed. Check 'optimization_comparison.png' and 'learning_dynamics.png'")