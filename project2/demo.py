import numpy as np
from newton import NewtonMethod
from conjugate_gradient import ConjugateGradient
from test_functions import QuadraticFunction, RosenbrockFunction


def demo_newton_method():  # 演示牛顿法
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    quad_func = QuadraticFunction(A, b)
    
    newton_opt = NewtonMethod(verbose=True, line_search=True)
    result = newton_opt.minimize(quad_func, np.array([0, 0]))
    
    print(f"最优解: {result.x}")
    print(f"理论最优解: {np.linalg.solve(A, b)}")
    print(f"最优值: {result.fun:.6e}")
    print(f"迭代次数: {result.niter}")
    print(f"是否成功: {result.success}")

def demo_conjugate_gradient():  # 演示共轭梯度法
    rosenbrock_func = RosenbrockFunction()
    cg_opt = ConjugateGradient(verbose=True, max_iter=1000)
    result = cg_opt.minimize(rosenbrock_func, np.array([-1.5, 2.0]))

    print(f"最优解: {result.x}")
    print(f"理论最优解: [1, 1]")
    print(f"最优值: {result.fun:.6e}")
    print(f"迭代次数: {result.niter}")
    print(f"是否成功: {result.success}")

if __name__ == "__main__":
    demo_newton_method()
    demo_conjugate_gradient()