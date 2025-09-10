import inspect
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义损失函数
def loss_function(params):
    x, y = params
    return x**2 + y**2

# 存储每种优化算法的损失历史
histories = {}

# 定义优化算法列表和对应的学习率
optimizer_configs = {
    'AdaGrad': {'optimizer': optim.Adagrad, 'lr': 0.5},
    'RMSProp': {'optimizer': optim.RMSprop, 'lr': 0.5, 'alpha': 0.9, 'eps': 1e-8},
    'Adam': {'optimizer': optim.Adam, 'lr': 0.5}
}

# 使用torch库的优化算法进行优化
for name, config in optimizer_configs.items():
    # 每次重新初始化参数
    params = torch.tensor([10.0, 10.0], requires_grad=True)
    
    # 创建优化器
    if name == 'RMSProp':
        optimizer = config['optimizer']([params], lr=config['lr'], alpha=config['alpha'], eps=config['eps'])
    else:
        optimizer = config['optimizer']([params], lr=config['lr'])
    
    losses = []
    steps = 120  # 优化步数
    
    for i in range(steps):
        optimizer.zero_grad()
        loss = loss_function(params)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    histories[name] = losses

# 绘制损失曲线
plt.figure(figsize=(12, 8))
for name, losses in histories.items():
    plt.plot(losses, label=name)

plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Comparison of Optimization Algorithms on f(x,y)=x^2+y^2')
plt.legend()
plt.grid(True)
savaPath = inspect.getfile(loss_function) + '/../_torch库的三种算法执行结果.png'
print(savaPath)
plt.savefig(savaPath, dpi=300, bbox_inches='tight')
plt.show()


# 打印最终结果
print("\nFinal Results:")
for name, losses in histories.items():
    print(f"{name}: Final loss = {losses[-1]:.8f}")

