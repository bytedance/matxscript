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
import matx
from typing import List, Dict, Set


class TestContainerComprehension(unittest.TestCase):

    def test_list_comp(self):
        def test_list_comp_nested(b: List[int]) -> List[int]:
            b = [j for i in b if i % 2 != 0 for j in range(i)]
            return b

        input_b = [1, 2, 3]
        py_ret = test_list_comp_nested(input_b)
        print(py_ret)
        tx_ret = matx.script(test_list_comp_nested)(input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

        def test_list_comp_with_if_expr(b: List[int]) -> List[int]:
            b = [1 if i > 2 else 0 for i in range(5, 10)]
            return b

        input_b = [1, 2, 3]
        py_ret = test_list_comp_nested(input_b)
        print(py_ret)
        tx_ret = matx.script(test_list_comp_nested)(input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_set_comp(self):
        def test_set_comp_nested(b: Set[int]) -> Set[int]:
            b = {j for i in b if i % 2 != 0 for j in range(i)}
            return b

        input_b = {1, 2, 3}
        py_ret = test_set_comp_nested(input_b)
        print(py_ret)
        tx_ret = matx.script(test_set_comp_nested)(input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_dict_comp(self):
        def test_dict_comp_nested(b: List[int]) -> Dict[int, int]:
            c = {j: j for i in b if i % 2 != 0 for j in range(i)}
            return c

        input_b = [1, 2, 3]
        py_ret = test_dict_comp_nested(input_b)
        print(py_ret)
        tx_ret = matx.script(test_dict_comp_nested)(input_b)
        print(tx_ret)
        self.assertEqual(py_ret, tx_ret)

    def test_capture_handle_var(self):
        def func1(i: int) -> int:
            return i + 1

        def my_func() -> List:
            return [func1(i) for i in range(1)]

        class MyClass:

            def __init__(self):
                self.a: int = 1

            def my_foo(self) -> List:
                return [func1(self.a) for i in range(1)]

        matx.script(my_func)()
        matx.script(MyClass)().my_foo()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
