"""
Tests for the example package.
"""

import example


def test_fibonacci():
    """
    Test the fibonacci function.
    """

    assert example.fibonacci(0) == 0
    assert example.fibonacci(1) == 1
    assert example.fibonacci(2) == 1
    assert example.fibonacci(3) == 2
    assert example.fibonacci(4) == 3
    assert example.fibonacci(5) == 5
    assert example.fibonacci(6) == 8
    assert example.fibonacci(7) == 13
