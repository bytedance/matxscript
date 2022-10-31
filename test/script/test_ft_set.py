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
from typing import List, Dict, Set, Tuple
import matx
from matx import FTSet


class TestFTSet(unittest.TestCase):

    def test_normal_ft_set(self):
        def test_ft_set_of_int() -> None:
            a: FTSet[int] = set()

        def test_ft_set_of_float() -> None:
            a: FTSet[float] = set()

        def test_ft_set_of_bool() -> None:
            a: FTSet[bool] = set()

        def test_ft_set_of_str() -> None:
            a: FTSet[str] = set()

        def test_ft_set_of_bytes() -> None:
            a: FTSet[bytes] = set()

        matx.script(test_ft_set_of_int)
        matx.script(test_ft_set_of_float)
        matx.script(test_ft_set_of_bool)
        matx.script(test_ft_set_of_str)
        matx.script(test_ft_set_of_bytes)

    def test_nested_ft_set(self):
        def test_list_of_ft_set() -> None:
            a: List[FTSet[int]] = []

        def test_set_of_ft_set() -> None:
            a: Set[FTSet[int]] = set()

        def test_dict_of_ft_set() -> None:
            a: Dict[int, FTSet[int]] = {}

        def test_tuple_of_ft_set() -> None:
            a: Tuple[int, FTSet[int]] = (0, {0})

        def test_ft_set_of_ft_set() -> None:
            a: FTSet[FTSet[int]] = set()

        def test_return_ft_set() -> FTSet[int]:
            return set()

        def test_arg_ft_set(a: FTSet[int]) -> None:
            return None

        def test_entry(py_func, *args):
            py_ret = py_func(*args)
            tx_func = matx.script(py_func)
            tx_ret = tx_func(*args)
            print(f"test func: {py_func}")
            print("py_ret: ", py_ret)
            print("tx_ret: ", tx_ret)
            self.assertEqual(py_ret, tx_ret)
            del tx_ret
            del tx_func

        test_entry(test_list_of_ft_set)
        test_entry(test_set_of_ft_set)
        test_entry(test_dict_of_ft_set)
        test_entry(test_tuple_of_ft_set)
        test_entry(test_ft_set_of_ft_set)

        matx.script(test_return_ft_set)
        matx.script(test_arg_ft_set)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
