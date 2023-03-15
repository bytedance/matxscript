#  Copyright 2023 ByteDance Ltd. and/or its affiliates.
#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import itertools
from typing import Callable, Dict, Tuple, Type, Union, Set

from ..base_op import KernelBaseOp

MethodType = Callable[..., KernelBaseOp]


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


class OpRegistry:
    # lhs_type rhs_type op_name
    _bin_op_repo: Dict[Tuple[str, str, str], MethodType] = {}
    _unary_op_repo: Dict[Tuple[str, str], MethodType] = {}
    _universal_func_repo: Dict[str, MethodType] = {}

    @classmethod
    def add_bin_op(cls,
                   op: MethodType,
                   left_type: str,
                   right_type: str,
                   op_name: str):
        cls._bin_op_repo[(op_name, left_type, right_type)] = op

    @classmethod
    def get_bin_op(cls, left_type, right_type, op_name: str):
        left_types = _get_all_bases(left_type.__class__)
        right_types = _get_all_bases(right_type.__class__)
        for left_t, right_t in itertools.product(left_types, right_types):
            key = (op_name, left_t, right_t)
            if key in cls._bin_op_repo:
                return cls._bin_op_repo.get(key)  # 'Array' 'Array' 'Add'
        return None
