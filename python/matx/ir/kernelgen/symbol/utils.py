import sympy


def is_symbol(x):
    return isinstance(x, sympy.Basic)


def equals(x, y):
    # https://stackoverflow.com/questions/37112738/sympy-comparing-expressions
    return sympy.simplify(x - y) == 0


def simplify(x):
    return sympy.simplify(x)
