from .base import ObjectiveFunction, Optimizer, OptimizationResult
from .newton import NewtonMethod
from .conjugate_gradient import ConjugateGradient
from .test_functions import QuadraticFunction, RosenbrockFunction
from .line_search import backtracking_line_search, strong_wolfe_line_search
from .demo import demo_newton_method, demo_conjugate_gradient

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = [
    'ObjectiveFunction', 'Optimizer', 'OptimizationResult',
    'NewtonMethod', 'ConjugateGradient',
    'QuadraticFunction', 'RosenbrockFunction',
    'backtracking_line_search', 'strong_wolfe_line_search',
    'demo_newton_method', 'demo_conjugate_gradient'
]