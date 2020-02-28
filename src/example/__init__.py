"""
An example package.
"""


def fibonacci(k: int) -> int:
    """
    Get the k-th fibonacci number.
    """

    first = 0
    second = 1

    for _ in range(k):
        first, second = second, first + second

    return first
