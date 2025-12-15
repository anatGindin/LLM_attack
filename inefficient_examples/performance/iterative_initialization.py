# HumanEval/24 by DeepSeek-Coder-6.7B
# he for loop can be optimized by starting from √n instead of n-1, since no divisor of n reater than √n exists, except for n itself


def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    for i in range(n - 1, 0, -1):
        if n % i == 0:
            return i