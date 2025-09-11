"""
优化算法库

一个包含多种优化算法的Python库，用于数值优化问题的求解。
包括梯度下降法、牛顿法、BFGS、DFP和共轭梯度法等优化算法。
"""

from .core import BaseOptimizer
from .optimizers import (
    GradientDescent, NewtonMethod, BFGS, DFP, ConjugateGradient
)
from .test_functions import quadratic_function, simple_quadratic, rosenbrock_function
from .utils import (
    plot_optimization_2d, plot_optimization_3d, plot_convergence,
    plot_interactive_optimization, plot_algorithm_comparison
)
from .benchmarks import Benchmark, run_quadratic_benchmark, run_rosenbrock_benchmark

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'BaseOptimizer',
    'GradientDescent',
    'NewtonMethod',
    'BFGS',
    'DFP',
    'ConjugateGradient',
    'quadratic_function',
    'simple_quadratic',
    'rosenbrock_function',
    'plot_optimization_2d',
    'plot_optimization_3d',
    'plot_convergence',
    'plot_interactive_optimization',
    'plot_algorithm_comparison',
    'Benchmark',
    'run_quadratic_benchmark',
    'run_rosenbrock_benchmark'
]