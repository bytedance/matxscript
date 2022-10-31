# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import unittest
from typing import List, Dict, Tuple, Any
import matx
from matx import FTList, FTDict


class TestFTNested(unittest.TestCase):

    def _test_entry(self, py_func, *args):
        py_ret = py_func(*args)
        tx_ret = matx.script(py_func)(*args)
        print(f"test func: {py_func}")
        print("py_ret: ", py_ret)
        print("tx_ret: ", tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_assignment(self):
        def outer_is_list() -> Any:
            l: List[FTList[int]] = [[], []]
            l[0] = [1, 2]
            return l

        def outer_is_ft_list() -> Any:
            l: FTList[FTList[int]] = [[], []]
            l[0] = [1, 2]
            return l

        def outer_is_dict() -> Any:
            d: Dict[str, FTList[int]] = {}
            d['123'] = [4, 5, 6]
            return d

        def outer_is_ft_dict() -> Any:
            d: FTDict[str, FTList[int]] = {}
            d['123'] = [4, 5, 6]
            return d

        # self._test_entry(outer_is_list)
        # TODO: BUG: __equal__ for List[FTList[int]] and python list
        self._test_entry(outer_is_ft_list)
        # self._test_entry(outer_is_dict)
        # TODO: BUG: __equal__ for Dict[str, FTList[int]] and python dict
        self._test_entry(outer_is_ft_dict)

    def test_as_func_arg(self):
        def func(l1: FTList[int], l2: List[int]) -> int:
            ret = 0
            for v in l1:
                ret += v
            for v in l2:
                ret += v
            return ret

        def wrapper() -> int:
            return func([1, 2, 3], [4, 5, 6])

        self._test_entry(wrapper)

    def test_as_class_init_arg(self):
        class MyClass:
            def __init__(self, l: FTList[int]) -> None:
                self.l: FTList[int] = l

            def func(self) -> int:
                ret = 0
                for v in self.l:
                    ret += v
                return ret

        def wrapper() -> int:
            c = MyClass([1, 2, 3])
            return c.func()

        self._test_entry(wrapper)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
