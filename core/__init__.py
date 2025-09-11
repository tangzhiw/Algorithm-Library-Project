"""
核心模块

包含优化器基类和线搜索算法。
"""

from .base_optimizer import BaseOptimizer
from .line_search import backtracking_line_search, wolfe_line_search

__all__ = [
    'BaseOptimizer',
    'backtracking_line_search',
    'wolfe_line_search'
]