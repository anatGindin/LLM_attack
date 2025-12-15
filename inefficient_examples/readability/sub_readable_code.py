# HumanEval/6 by DeepSeek-Coder-6.7B
# the combination of list comprehension, map, and lambda functions, along with a for loop and multiple method calls, makes the code hard to read and requires additional effort to comprehend

from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
    return [max(map(lambda x: x.count('('), group.split())) for group in paren_string.split()]