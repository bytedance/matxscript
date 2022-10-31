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
from matx import FTDict


class TestFTSet(unittest.TestCase):

    def test_normal_ft_dict(self):
        def test_ft_dict_of_int_int() -> None:
            a: FTDict[int, int] = {}

        def test_ft_dict_of_bool_bool() -> None:
            a: FTDict[bool, bool] = {}

        def test_ft_dict_of_str_str() -> None:
            a: FTDict[str, str] = {}

        def test_ft_dict_of_bytes_bytes() -> None:
            a: FTDict[bytes, bytes] = {}

        matx.script(test_ft_dict_of_int_int)
        matx.script(test_ft_dict_of_bool_bool)
        matx.script(test_ft_dict_of_str_str)
        matx.script(test_ft_dict_of_bytes_bytes)

    def test_nested_ft_dict(self):
        def test_list_of_ft_dict() -> None:
            a: List[FTDict[int, int]] = []

        def test_set_of_ft_dict() -> None:
            a: Set[FTDict[int, int]] = set()

        def test_dict_of_ft_dict() -> None:
            a: Dict[int, FTDict[int, int]] = {}

        def test_tuple_of_ft_dict() -> None:
            a: Tuple[int, FTDict[int, int]] = (0, {})

        def test_ft_dict_of_ft_dict() -> None:
            a: FTDict[int, FTDict[int, int]] = {}

        def test_return_ft_dict() -> FTDict[int, int]:
            return {}

        def test_arg_ft_dict(a: FTDict[int, int]) -> None:
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

        test_entry(test_list_of_ft_dict)
        test_entry(test_set_of_ft_dict)
        test_entry(test_dict_of_ft_dict)
        test_entry(test_tuple_of_ft_dict)
        test_entry(test_ft_dict_of_ft_dict)

        matx.script(test_return_ft_dict)
        matx.script(test_arg_ft_dict)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
