import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Set

MethodType = Callable[..., Tuple[str]]  # todo why tuple string


def _get_all_bases(class_or_name: Union[str, Type]) -> Set[str]:
    """
    Returns a list of the current class name and all its base classes.

    :param class_or_name: A class type or a class name.
    :return: A list of strings representing class names if a type was given, or a list with a single
             string if a string was given. The list is given in reverse order, with subclasses preceding
             superclasses.
    """
    if isinstance(class_or_name, str):
        return {class_or_name}

    classes = {class_or_name.__name__}
    for base in class_or_name.__bases__:
        classes.union(_get_all_bases(base))

    return classes  # todo keep orders


class OpReplacementRepo:
    _bin_op_repo: Dict[Tuple[str, str, str], MethodType] = {}
    _unary_op_repo: Dict[Tuple[str, str], MethodType] = {}
    _universal_func_repo: Dict[str, MethodType] = {}

    @classmethod
    def add_bin_operator(cls,
                         func: Callable[[Any, Any, str, str], Tuple[str]],
                         left_type: str,
                         right_type: str,
                         op_name: str):
        cls._bin_op_repo[(op_name, left_type, right_type)] = func

    @classmethod
    def get_bin_operator(cls, left_type: str, right_type: str, op_name: str):
        left_types = _get_all_bases(left_type)
        right_types = _get_all_bases(right_type)
        for left_t, right_t in itertools.product(left_types, right_types):
            key = (op_name, left_t, right_t)
            if key in cls._bin_op_repo:
                return cls._bin_op_repo.get(key)  # 'Array' 'Array' 'Add'
        return None
