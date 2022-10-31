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

from typing import Tuple, Any
import unittest
import matx


class TestContainerTuple(unittest.TestCase):
    def test_len_tuple2(self):
        def len_tuple(container: Tuple[int, int]) -> int:
            return len(container)

        scripted_len_tuple = matx.script(len_tuple)

        t2 = (1, 2)
        self.assertEqual(scripted_len_tuple(t2), 2)

    def test_len_tuple3(self):
        def len_tuple(container: Tuple[int, int, int]) -> int:
            return len(container)

        scripted_len_tuple = matx.script(len_tuple)
        t3 = (1, 2, 3)
        self.assertEqual(scripted_len_tuple(t3), 3)

    def test_tuple_repeat(self):
        def tuple_repeat(times: int) -> int:
            origin = (1, 2, 4)
            b = origin * times
            return len(origin) + len(b) + b[5]

        tuple_repeat_op = matx.script(tuple_repeat)
        self.assertEqual(tuple_repeat(3), 16)
        self.assertEqual(tuple_repeat_op(3), 16)

        def tuple_repeat_object_type_times() -> int:
            origin = (1, 2, 4)
            b = origin * origin[1]
            return len(origin) + len(b) + b[5]

        tuple_repeat_object_type_times_op = matx.script(tuple_repeat_object_type_times)
        self.assertEqual(tuple_repeat_object_type_times(), 13)
        self.assertEqual(tuple_repeat_object_type_times_op(), 13)

    def test_generic_repeat(self):
        def generic_repeat(times: int) -> int:
            origin_list = [(1, 2, 4)]
            origin = origin_list[0]
            b = origin * times
            return len(origin) + len(b) + b[5]

        generic_repeat_op = matx.script(generic_repeat)
        self.assertEqual(generic_repeat(3), 16)
        self.assertEqual(generic_repeat_op(3), 16)

    def test_tuple_in(self):
        def tuple_in(x: Any) -> bool:
            if 2 in x:
                return True
            else:
                return False

        tuple_in_op = matx.script(tuple_in)
        self.assertEqual(tuple_in((1, 3, 5,)), False)
        self.assertEqual(tuple_in((1, 2, 3,)), True)
        self.assertEqual(tuple_in_op((1, 3, 5,)), False)
        self.assertEqual(tuple_in_op((1, 2, 3,)), True)

    def test_general_in(self):
        def general_in(x: Any) -> bool:
            if 2 in x:
                return True
            else:
                return False

        general_in_op = matx.script(general_in)
        self.assertEqual(general_in((1, 3, 5,)), False)
        self.assertEqual(general_in((1, 2, 3,)), True)
        self.assertEqual(general_in_op((1, 3, 5,)), False)
        self.assertEqual(general_in_op((1, 2, 3,)), True)

    def test_tuple_notin(self):
        def tuple_notin(x: Any) -> bool:
            if 2 not in x:
                return True
            else:
                return False

        tuple_notin_op = matx.script(tuple_notin)
        self.assertEqual(tuple_notin((1, 3, 5,)), True)
        self.assertEqual(tuple_notin((1, 2, 3,)), False)
        self.assertEqual(tuple_notin_op((1, 3, 5,)), True)
        self.assertEqual(tuple_notin_op((1, 2, 3,)), False)

    def test_general_notin(self):
        def general_notin(x: Any) -> bool:
            if 2 not in x:
                return True
            else:
                return False

        general_notin_op = matx.script(general_notin)
        self.assertEqual(general_notin((1, 3, 5,)), True)
        self.assertEqual(general_notin((1, 2, 3,)), False)
        self.assertEqual(general_notin_op((1, 3, 5,)), True)
        self.assertEqual(general_notin_op((1, 2, 3,)), False)

    def test_tuple_subscript(self):

        def tuple_subscript(a: Any) -> Any:
            return a[1:3]

        tuple_subscript_op = matx.script(tuple_subscript)
        self.assertEqual(tuple_subscript((1, 2, 3, 4,)), (2, 3,))
        self.assertEqual(tuple_subscript_op((1, 2, 3, 4,)), (2, 3,))

    def test_general_subscript(self):

        def general_subscript(a: Any) -> Any:
            return a[1:3]

        general_subscript_op = matx.script(general_subscript)
        self.assertEqual(general_subscript((1, 2, 3, 4,)), (2, 3,))
        self.assertEqual(general_subscript_op((1, 2, 3, 4,)), (2, 3,))

    def test_tuple_iterator(self):
        def tuple_iterator(a: Tuple[int, int, int]) -> Any:
            result = []
            for x in a:
                result.append(x)
            return result

        tuple_iterator_op = matx.script(tuple_iterator)
        data = (1, 3, 5,)
        self.assertEqual(list(data), tuple_iterator(data))
        self.assertEqual(list(data), tuple_iterator_op(data))

    def test_generic_iterator(self):
        def generic_iterator(a: Any) -> Any:
            result = []
            for x in a:
                result.append(x)
            return result

        generic_iterator_op = matx.script(generic_iterator)
        data = (1, 3, 5)
        self.assertEqual(list(data), generic_iterator(data))
        self.assertEqual(list(data), generic_iterator_op(data))

    def test_tuple_slice(self):
        def tuple_slice(a: Any) -> Any:
            slice_size = len(a) + 3
            return a[:slice_size]

        python_tuple = [1, 2, 3, 4, 5, 6]
        matx_tuple = matx.Tuple(*python_tuple)
        python_ret = python_tuple[:9]
        matx_ret = matx_tuple[:9]
        matx_script_slice = matx.script(tuple_slice)
        matx_script_ret = matx_script_slice(python_tuple)

        self.assertEqual(python_tuple, python_ret)
        self.assertEqual(python_tuple, matx_ret)
        self.assertEqual(python_tuple, matx_script_ret)

    def test_tuple_hash_equal(self):
        def tuple_explicit_equal() -> Any:
            tup1 = (1, 2, "2", b"h", ["h", 1, b"x"])
            tup2 = (1, 2, "2", b"h", ["h", 1, b"x"])
            return tup1 == tup2

        py_ret = tuple_explicit_equal()
        tx_ret = matx.script(tuple_explicit_equal)()
        self.assertEqual(py_ret, tx_ret)

        def tuple_any_equal() -> Any:
            tup1 = (1, 2, "2", b"h", ["h", 1, b"x"])
            tup2: Any = (1, 2, "2", b"h", ["h", 1, b"x"])
            tup3: Any = (1, 2, "2", b"h", ["h", 1, b"x"])
            return tup1 == tup2, tup2 == tup3

        py_ret = tuple_any_equal()
        tx_ret = matx.script(tuple_any_equal)()
        self.assertEqual(py_ret, tx_ret)

        def tuple_hash() -> Any:
            tup1 = (1, 2)
            tup2 = (1, 2)
            tup3: Any = (1, 2)
            d = dict()
            d[tup1] = 10
            return d[tup2] == 10, d[tup3] == 10

        py_ret = tuple_hash()
        tx_ret = matx.script(tuple_hash)()
        self.assertEqual(py_ret, tx_ret)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
