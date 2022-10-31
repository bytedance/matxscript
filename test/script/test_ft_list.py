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
import copy
import unittest
import random
from typing import List, Dict, Set, Tuple, Any
import matx
from matx import FTList


class TestFTList(unittest.TestCase):

    def test_print_ft_list(self):
        def test_print_ft_list_impl() -> None:
            a: FTList[int] = [1, 2, 3]
            print(a)
            b: FTList[str] = ["hello", "world"]
            print(b)
            c: FTList[bytes] = [b"hello", b"world"]
            print(c)

        matx.script(test_print_ft_list_impl)()

    def test_normal_ft_list(self):
        def test_ft_list_of_int() -> None:
            a: FTList[int] = []

        def test_ft_list_of_float() -> None:
            a: FTList[float] = []

        def test_ft_list_of_bool() -> None:
            a: FTList[bool] = []

        def test_ft_list_of_str() -> None:
            a: FTList[str] = []

        def test_ft_list_of_bytes() -> None:
            a: FTList[bytes] = []

        def test_ft_list_of_list() -> None:
            a: FTList[List[int]] = []

        def test_ft_list_of_set() -> None:
            a: FTList[Set[int]] = []

        def test_ft_list_of_dict() -> None:
            a: FTList[Dict[int, int]] = []

        def test_ft_list_of_tuple() -> None:
            a: FTList[Tuple[int, int]] = [(0, 0)]

        matx.script(test_ft_list_of_int)
        matx.script(test_ft_list_of_float)
        matx.script(test_ft_list_of_bool)
        matx.script(test_ft_list_of_str)
        matx.script(test_ft_list_of_bytes)
        matx.script(test_ft_list_of_list)
        matx.script(test_ft_list_of_set)
        matx.script(test_ft_list_of_dict)
        matx.script(test_ft_list_of_tuple)

    def test_nested_ft_list(self):
        def test_list_of_ft_list() -> Any:
            a: List[FTList[int]] = []
            return a

        def test_set_of_ft_list() -> Any:
            a: Set[FTList[int]] = set()
            return a

        def test_dict_of_ft_list() -> Any:
            a: Dict[int, FTList[int]] = {}
            return a

        def test_tuple_of_ft_list() -> Any:
            a: Tuple[int, FTList[int]] = (0, [0])
            return a

        def test_ft_list_of_ft_list() -> Any:
            a: FTList[FTList[int]] = []
            return None
            # return a

        def test_return_ft_list() -> FTList[int]:
            return []

        def test_arg_ft_list(a: FTList[int]) -> None:
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

        test_entry(test_list_of_ft_list)
        test_entry(test_set_of_ft_list)
        test_entry(test_dict_of_ft_list)
        test_entry(test_tuple_of_ft_list)
        test_entry(test_ft_list_of_ft_list)

        matx.script(test_return_ft_list)
        matx.script(test_arg_ft_list)

    def test_ft_list_basic_sort(self):
        def ft_list_basic_sort(a: List[int], e: List[int]) -> Any:
            b: FTList[int] = [k for k in a]
            b.sort()
            result = [b[k] == e[k] for k in range(len(b))]
            return result

        data = [random.randint(0, 100) for i in range(10)]
        sorted_data = sorted(data)
        py_ret = ft_list_basic_sort(data, sorted_data)
        tx_ft_list_basic_sort = matx.script(ft_list_basic_sort)
        tx_ret1 = tx_ft_list_basic_sort(data, sorted_data)
        self.assertSequenceEqual(py_ret, tx_ret1)

    def test_ft_list_reverse_sort(self):
        def ft_list_reverse_sort(a: List[int], e: List[int]) -> Any:
            b: FTList[int] = [k for k in a]
            b.sort(reverse=True)
            result = [b[k] == e[k] for k in range(len(b))]
            return result

        data = [random.randint(0, 100) for i in range(10)]
        sorted_data = sorted(data, reverse=True)
        py_ret = ft_list_reverse_sort(data, sorted_data)
        tx_ft_list_reverse_sort = matx.script(ft_list_reverse_sort)
        tx_ret1 = tx_ft_list_reverse_sort(data, sorted_data)
        self.assertSequenceEqual(py_ret, tx_ret1)

    def test_ft_list_key_sort(self):
        def my_key(x: int) -> int:
            return -x

        def ft_list_key_sort(a: List[int], e: List[int]) -> Any:
            b: FTList[int] = [k for k in a]
            b.sort(key=my_key)
            result = [b[k] == e[k] for k in range(len(b))]
            return result

        data = [random.randint(0, 100) for i in range(10)]
        sorted_data = sorted(data, key=my_key)
        py_ret = ft_list_key_sort(data, sorted_data)
        tx_ft_list_key_sort = matx.script(ft_list_key_sort)
        tx_ret1 = tx_ft_list_key_sort(data, sorted_data)
        self.assertSequenceEqual(py_ret, tx_ret1)

    def test_ftlist_index(self):

        def list_index_int(l: List[int], target: int) -> int:
            ft: FTList[int] = [k for k in l]
            i: int = ft.index(target)
            return i

        def list_index_with_start(l: List[int], target: int, start: int) -> int:
            ft: FTList[int] = [k for k in l]
            return ft.index(target, start)

        def list_index_with_start_end(l: List[int], target: int, start: int, end: int) -> int:
            ft: FTList[int] = [k for k in l]
            return ft.index(target, start, end)

        def list_index_with_start_end2(target: int, start: int, end: int) -> int:
            l: FTList[int] = [1, 2, 3, 4, 5]
            return l.index(target, start, end)

        list_index_int_op = matx.script(list_index_int)
        list_index_with_start_op = matx.script(list_index_with_start)
        list_index_with_start_end_op = matx.script(list_index_with_start_end)
        list_index_with_start_end_op2 = matx.script(list_index_with_start_end2)

        l_int: List = list([1, 2, 3, 4, 5])
        idx = list_index_int_op(l_int, 3)
        self.assertEqual(idx, 2)
        idx = list_index_int_op(l_int, 4)
        self.assertEqual(idx, 3)

        idx = list_index_with_start_op(l_int, 3, 0)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, 1)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, 2)
        self.assertEqual(idx, 2)
        idx = list_index_with_start_op(l_int, 3, -3)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, 3)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, 100)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_op(l_int, 3, -1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        idx = list_index_with_start_end_op(l_int, 3, 2, 3)
        self.assertEqual(idx, 2)

        idx = list_index_with_start_end_op(l_int, 3, -3, -1)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op(l_int, 3, 2, 2)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end(l_int, 3, 4, 1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        idx = list_index_with_start_end_op2(3, 2, 3)
        self.assertEqual(idx, 2)

        idx = list_index_with_start_end_op2(3, -3, -1)
        self.assertEqual(idx, 2)

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op2(3, 2, 2)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))

        with self.assertRaises(ValueError) as err:
            idx = list_index_with_start_end_op2(3, 4, 1)
            self.assertEqual("""ValueError: '3' is not in list""", str(err.exception))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
