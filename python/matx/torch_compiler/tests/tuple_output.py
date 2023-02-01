import matx

from typing import Tuple


@matx.script
def func(a: int, b: int) -> Tuple[int, int]:
    return a, b
