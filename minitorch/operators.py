"""
Mathematical operators for MiniTorch.

These form the foundation of all neural network operations.
You'll implement each function to understand how deep learning
frameworks handle basic mathematics.
"""

import math
from typing import Callable, Iterable


# TODO: Implement these functions in Task 0.1
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y  


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Less than comparison."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equality comparison."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close (within 1e-2)."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0  # Q8: What tolerance?


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    """ReLU activation function."""
    return max(0.0, x)


def log(x: float) -> float:
    """Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal function."""
    return 1.0 / x


def log_back(x: float, grad: float) -> float:
    """
    Gradient of log(x) times incoming gradient.
    Derivative of log(x) is 1/x.
    """
    return grad / x  # Q15: Divide by what?


def inv_back(x: float, grad: float) -> float:
    """
    Gradient of 1/x times incoming gradient.
    Derivative of 1/x is -1/x^2.
    """
    return -grad / (x * x)  # Q16: What squared?


def relu_back(x: float, grad: float) -> float:
    """Gradient of ReLU."""
    return grad if x > 0 else 0.0


# TODO: Implement these in Task 0.3
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function."""
    raise NotImplementedError("Implement in Task 0.3")


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith function."""
    raise NotImplementedError("Implement in Task 0.3")


def reduce(fn: Callable[[float, float], float], init: float) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function."""
    raise NotImplementedError("Implement in Task 0.3")


def sum(ls: Iterable[float]) -> float:
    """Sum using reduce."""
    raise NotImplementedError("Implement in Task 0.3")


def prod(ls: Iterable[float]) -> float:
    """Product using reduce."""
    raise NotImplementedError("Implement in Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate list using map."""
    raise NotImplementedError("Implement in Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add lists using zipWith."""
    raise NotImplementedError("Implement in Task 0.3")
