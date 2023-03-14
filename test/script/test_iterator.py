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

from typing import List, Dict, Set, Tuple
from typing import Any
import unittest
import matx
import traceback


class TestIterator(unittest.TestCase):
    def test_str_iter(self):
        def str_iter(a: str) -> None:
            for s in a:
                print(s)
            return None

        def byte_iter(a: bytes) -> None:
            for s in a:
                print(s)
            return None

        def generic_str_iter(a: Any) -> None:
            for s in a:
                print(s)
            return None

        def generic_byte_iter(a: Any) -> None:
            for s in a:
                print(s)
            return None

        matx.script(str_iter)("hello")
        matx.script(byte_iter)(b"hello")
        matx.script(generic_str_iter)("hello")
        matx.script(generic_byte_iter)(b"hello")

    def test_tuple_iter(self):
        def tuple_iter(a: Tuple[str, str]) -> None:
            for s in a:
                print(s)
            return None

        def generic_tuple_iter(a: Any) -> None:
            for s in a:
                print(s)
            return None

        matx.script(tuple_iter)(("hello", "hello"))
        matx.script(generic_tuple_iter)(("hello", "hello"))

    def test_list_builtin_iter(self):
        def builtin_list_iter(li: List) -> List:
            new_list = []
            for l in iter(li):
                new_list.append(l)
            return new_list

        original_list = [1, 2, 3, 4, 5]
        python_iter_result = builtin_list_iter(original_list)
        matx_iter_result = matx.script(builtin_list_iter)(original_list)
        self.assertListEqual(python_iter_result, list(matx_iter_result))

    def test_list_reversed_iter(self):
        def builtin_list_reversed(li: List) -> List:
            new_list = []
            for l in reversed(li):
                new_list.append(l)
            return new_list

        original_list = [1, 2, 3, 4, 5]
        python_reversed_result = builtin_list_reversed(original_list)
        matx_reversed_result = matx.script(builtin_list_reversed)(original_list)
        self.assertListEqual(python_reversed_result, list(matx_reversed_result))

    # tuple's iter is not implemented
    def test_tuple_iter_exception(self):
        def builtin_tuple_iter(li: Tuple[int, int]) -> List:
            new_list = []
            for l in iter(li):
                new_list.append(l)
            return new_list

        original_tuple = (1, 2)
        try:
            matx_iter_result = matx.script(builtin_tuple_iter)(original_tuple)
        except Exception as e:
            traceback.print_exc()

    # tuple's reversed is not implemented
    def test_tuple_reversed_exception(self):
        def builtin_tuple_reversed(li: Tuple[int, int]) -> List:
            new_list = []
            for l in reversed(li):
                new_list.append(l)
            return new_list

        original_tuple = (1, 2)
        try:
            matx_reversed_result = matx.script(builtin_tuple_reversed)(original_tuple)
        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
