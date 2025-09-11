import numpy as np
import time
import matplotlib.pyplot as plt
from problems import LogisticProblem
from optimizers import GDOptimizer, SGDOptimizer, MiniBatchOptimizer
from baseline import pytorch_logistic_regression

# 中文字体与负号兼容
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def benchmark():
    rng = np.random.default_rng(42)

    # ======== 构造数据 (float32) ========
    n, d = 5000, 50
    X = rng.standard_normal((n, d), dtype=np.float32)
    true_w = rng.standard_normal(d, dtype=np.float32)
    logits = X @ true_w
    y = (logits + 0.1 * rng.standard_normal(n, dtype=np.float32) > 0).astype(np.float32)

    prob = LogisticProblem(X, y)
    w0 = np.zeros(prob.dim(), dtype=np.float32)

    # ======== 训练 ========
    configs = {
        "GD": GDOptimizer(lr=0.2, weight_decay=1e-4),
        "SGD": SGDOptimizer(lr=0.05, momentum=0.9, weight_decay=1e-4, seed=42),
        "MiniBatch": MiniBatchOptimizer(lr=0.1, batch_size=256, momentum=0.9, weight_decay=1e-4, lr_decay=0.02, seed=42),
    }

    results = {}
    for name, opt in configs.items():
        w = w0.copy()
        t0 = time.perf_counter()
        hist = opt.run(w, prob, max_iters=500, log_every=10)
        elapsed = time.perf_counter() - t0
        results[name] = (elapsed, hist)

    # ======== PyTorch baseline ========
    t0 = time.perf_counter()
    torch_losses = pytorch_logistic_regression(X, y, lr=0.1, batch_size=256, max_epochs=10)
    torch_time = time.perf_counter() - t0
    results["PyTorch-MiniBatch"] = (torch_time, torch_losses)

    # ======== 画图 ========
    plt.figure(figsize=(8, 6))
    for name, (t, hist) in results.items():
        plt.plot(hist, label=f"{name} (time={t:.2f}s)")
    plt.xlabel("Iterations/10 or Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("自研算法库 vs PyTorch baseline")
    plt.tight_layout()
    plt.savefig(str((Path('.').resolve() / 'curve.png')))
    # plt.show()  # 如需交互显示可解除注释

if __name__ == "__main__":
    from pathlib import Path
    benchmark()
